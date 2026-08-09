# -*- coding: utf-8 -*-
"""
Round-trip сохранения и загрузки для всех нейросетевых архитектур.

Для каждой модели выполняется полный цикл:
    build → forward pass → save → load → predict
и сравниваются формы и численные значения прогноза до и после загрузки.

Отдельно проверяется контракт инференса. Раньше predict_from_bundle подавал
вход формы (1, 48, 1) модели, ожидающей (None, 48, 26). Часть архитектур
принимала такой вход за счёт broadcasting и возвращала формально корректный,
но бессмысленный прогноз — молчаливая ошибка, которую невозможно заметить по
метрикам. Инференс обязан строить те же N_FEATURES признаков в том же порядке,
что и обучение, и отвергать вход другой ширины.
"""

import numpy as np
import pytest
import tensorflow as tf

from config import Config
from data.generator import generate_smartgrid_data
from data.preprocessing import prepare_data
from utils.deployment import (
    export_model_bundle, load_model_bundle, predict_from_bundle,
)

N_FEATURES = 26
HISTORY = 48
HORIZON = 24


@pytest.fixture(scope="module")
def tiny_data():
    df = generate_smartgrid_data(days=25, households=30, seed=13,
                                 industrial_loads=2, city_districts=2)
    data = prepare_data(df, history_length=HISTORY, forecast_horizon=HORIZON)
    return df, data


def _build(name: str):
    """Компактные версии архитектур: проверяется сериализация, не качество."""
    from models.lstm import build_lstm_model
    from models.transformer import build_vanilla_transformer, build_patchtst

    if name == "LSTM":
        return build_lstm_model(
            history_length=HISTORY, forecast_horizon=HORIZON, n_features=N_FEATURES,
            lstm_units_1=8, tcn_filters=4, attn_heads=1, seasonal_blend_init=0.35,
        )
    if name == "VanillaTransformer":
        return build_vanilla_transformer(
            history_length=HISTORY, forecast_horizon=HORIZON, n_features=N_FEATURES,
            d_model=16, num_heads=2, num_layers=1, dff=32,
            use_seasonal_residual=True, seasonal_blend_init=0.40, huber_delta=0.05,
        )
    if name == "PatchTST":
        return build_patchtst(
            history_length=HISTORY, forecast_horizon=HORIZON, patch_len=8, stride=4,
            n_features=N_FEATURES, d_model=16, num_heads=2, num_layers=1, dff=32,
            use_revin=True,
        )
    raise ValueError(name)


@pytest.mark.parametrize("model_name", ["LSTM", "VanillaTransformer", "PatchTST"])
def test_save_load_predict_roundtrip(model_name, tiny_data, tmp_path):
    """
    Прогноз после загрузки должен совпадать с прогнозом исходной модели.

    Расхождение означало бы, что часть состояния (веса нестандартных слоёв,
    параметры RevIN, коэффициент сезонного скипа) не пережила сериализацию.
    """
    df, data = tiny_data
    model = _build(model_name)

    x = data["X_test"][:4]
    assert x.shape[1:] == (HISTORY, N_FEATURES)

    before = model.predict(x, verbose=0)
    assert before.shape == (4, HORIZON)
    assert np.isfinite(before).all()

    bundle_dir = export_model_bundle(
        model, data,
        {"HISTORY_LENGTH": HISTORY, "FORECAST_HORIZON": HORIZON,
         "N_FEATURES": N_FEATURES, "model_name": model_name},
        export_dir=str(tmp_path), model_name=model_name,
    )

    bundle = load_model_bundle(bundle_dir)
    after = bundle["model"].predict(x, verbose=0)

    assert after.shape == before.shape, f"{model_name}: форма изменилась"
    assert np.allclose(before, after, rtol=1e-5, atol=1e-5), (
        f"{model_name}: прогноз после загрузки отличается — часть весов потеряна"
    )


@pytest.mark.parametrize("model_name", ["LSTM", "VanillaTransformer", "PatchTST"])
def test_bundle_contains_full_feature_schema(model_name, tiny_data, tmp_path):
    """Бандл должен содержать все скалеры и схему признаков."""
    df, data = tiny_data
    bundle_dir = export_model_bundle(
        _build(model_name), data,
        {"HISTORY_LENGTH": HISTORY, "FORECAST_HORIZON": HORIZON,
         "N_FEATURES": N_FEATURES, "model_name": model_name},
        export_dir=str(tmp_path), model_name=model_name,
    )
    bundle = load_model_bundle(bundle_dir)

    required = ("scaler", "temp_scaler", "humidity_scaler", "wind_scaler",
                "rolling_std_scaler", "cloud_scaler", "ev_scaler", "solar_scaler")
    for key in required:
        assert bundle["scalers"].get(key) is not None, f"в бандле нет {key}"
    assert bundle["scalers"].get("temp_sq_max") is not None
    assert bundle["config"]["N_FEATURES"] == N_FEATURES
    assert bundle["config"]["HISTORY_LENGTH"] == HISTORY


@pytest.mark.parametrize("model_name", ["LSTM", "VanillaTransformer", "PatchTST"])
def test_inference_reproduces_training_preprocessing(model_name, tiny_data, tmp_path):
    """
    predict_from_bundle обязан построить те же 26 признаков в том же порядке.

    Сравнение идёт с прямым прогоном модели по матрице, собранной штатным
    препроцессингом: расхождение означало бы, что инференс и обучение видят
    разные признаки.
    """
    from data.preprocessing import (
        _add_lag_columns, _build_feature_matrix, inverse_scale,
    )

    df, data = tiny_data
    model = _build(model_name)
    bundle_dir = export_model_bundle(
        model, data,
        {"HISTORY_LENGTH": HISTORY, "FORECAST_HORIZON": HORIZON,
         "N_FEATURES": N_FEATURES, "model_name": model_name},
        export_dir=str(tmp_path), model_name=model_name,
    )
    bundle = load_model_bundle(bundle_dir)

    recent = df.tail(200)
    forecast = predict_from_bundle(bundle, recent)

    features = _build_feature_matrix(
        _add_lag_columns(recent.copy()).iloc[-HISTORY:],
        cons_scaler=data["scaler"], temp_scaler=data["temp_scaler"],
        humidity_scaler=data["humidity_scaler"], wind_scaler=data["wind_scaler"],
        rolling_std_scaler=data["rolling_std_scaler"],
        cloud_scaler=data["cloud_scaler"],
        ev_scaler=data["ev_scaler"], solar_scaler=data["solar_scaler"],
        temp_sq_max=data["temp_sq_max"],
    )
    assert features.shape == (HISTORY, N_FEATURES), (
        "инференс должен собирать полную матрицу признаков"
    )

    expected = inverse_scale(
        data["scaler"], model.predict(features[np.newaxis], verbose=0)
    )[0]

    assert forecast.shape == (HORIZON,)
    assert np.allclose(forecast, expected, rtol=1e-4, atol=1e-2), (
        f"{model_name}: инференс не воспроизводит обучающий препроцессинг"
    )


@pytest.mark.parametrize("model_name", ["LSTM", "VanillaTransformer", "PatchTST"])
def test_single_channel_input_is_rejected(model_name, tiny_data, tmp_path):
    """
    Вход формы (1, 48, 1) должен отвергаться, а не проходить по broadcasting.

    Именно так выглядел прежний дефект: PatchTST принимал одноканальный вход
    и возвращал численно правдоподобный, но бессмысленный прогноз.
    """
    df, data = tiny_data
    model = _build(model_name)
    bundle_dir = export_model_bundle(
        model, data,
        {"HISTORY_LENGTH": HISTORY, "FORECAST_HORIZON": HORIZON,
         "N_FEATURES": N_FEATURES, "model_name": model_name},
        export_dir=str(tmp_path), model_name=model_name,
    )
    bundle = load_model_bundle(bundle_dir)

    # DataFrame только с потреблением: остальных ковариат нет.
    crippled = df.tail(200)[["timestamp", "consumption", "hour", "weekday",
                             "is_weekend", "is_holiday"]].copy()

    # Препроцессинг подставит нули вместо отсутствующих ковариат, но ширина
    # матрицы обязана остаться равной N_FEATURES — иначе модель молча съест
    # неверный вход.
    forecast = predict_from_bundle(bundle, crippled)
    assert forecast.shape == (HORIZON,)

    # Одноканальный тензор обязан быть отвергнут проверкой формы.
    # Сам Keras его пропускает: PatchTST транслирует последнюю размерность и
    # возвращает численно правдоподобный, но бессмысленный прогноз.
    from utils.deployment import assert_input_matches_model

    bad_input = np.zeros((1, HISTORY, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="признаков"):
        assert_input_matches_model(bundle["model"], bad_input,
                                   expected_features=N_FEATURES)

    # Неверная длина окна тоже должна отсекаться.
    with pytest.raises(ValueError, match="Длина окна"):
        assert_input_matches_model(
            bundle["model"], np.zeros((1, HISTORY // 2, N_FEATURES), dtype=np.float32),
            expected_features=N_FEATURES,
        )

    # Корректный вход проходит без исключения.
    assert_input_matches_model(
        bundle["model"], np.zeros((1, HISTORY, N_FEATURES), dtype=np.float32),
        expected_features=N_FEATURES,
    )


def test_lstm_has_no_lambda_layers():
    """
    В LSTM не должно остаться слоёв Lambda с Python-функцией внутри.

    Keras отказывается безопасно десериализовать такие слои, из-за чего
    сохранённую модель невозможно загрузить обратно.
    """
    model = _build("LSTM")
    lambda_layers = [l.name for l in model.layers
                     if isinstance(l, tf.keras.layers.Lambda)]
    assert not lambda_layers, f"остались Lambda-слои: {lambda_layers}"


def test_custom_objects_cover_all_custom_layers():
    """
    Словарь custom_objects обязан покрывать все нестандартные слои всех моделей.

    Пропуск хотя бы одного делает загрузку невозможной, причём ошибка
    проявится только при попытке инференса.
    """
    import inspect as _inspect
    from utils import deployment

    src = _inspect.getsource(deployment.load_model_bundle)

    standard = set(dir(tf.keras.layers))
    for name in ("LSTM", "VanillaTransformer", "PatchTST"):
        for layer in _build(name).layers:
            cls = type(layer).__name__
            if cls in standard:
                continue
            assert f'"{cls}"' in src, (
                f"{cls} (модель {name}) отсутствует в custom_objects"
            )


def test_sinusoidal_pe_deserializes():
    """SinusoidalPE должен восстанавливаться из конфигурации без потери буфера."""
    from models.transformer import SinusoidalPE

    layer = SinusoidalPE(16, max_len=64, name="pe")
    x = np.random.rand(2, HISTORY, 16).astype(np.float32)
    out_before = layer(tf.constant(x)).numpy()

    restored = SinusoidalPE.from_config(layer.get_config())
    out_after = restored(tf.constant(x)).numpy()

    assert out_before.shape == out_after.shape
    assert np.allclose(out_before, out_after, atol=1e-6), (
        "позиционное кодирование изменилось после восстановления из конфига"
    )
