# -*- coding: utf-8 -*-
"""
Тесты вероятностных моделей.

Проверяется то, что отличает вероятностный прогноз от точечного: уровни идут в
правильном порядке, обучение действительно минимизирует pinball, а не
маскируется под него, и модель воспроизводит известное распределение, а не
просто выдаёт три близких числа.
"""

import numpy as np
import pytest
import tensorflow as tf

from data.panel import generate_panel_data
from data.panel_preprocessing import prepare_panel_data
from models.panel_trainer import make_batch
from models.quantile_models import (
    DEFAULT_QUANTILES, PinballLoss, QuantileNaive, build_quantile_dlinear,
)
from utils.quantile_metrics import coverage, crossing_rate, pinball_loss


@pytest.fixture(scope="module")
def panel_data():
    df, specs = generate_panel_data(days=80, n_cities=1, feeders_per_city=3, seed=6)
    return prepare_panel_data(df, specs, history_length=48, forecast_horizon=24, seed=6)


# ══════════════════════════════════════════════════════════════════════════════
# ФУНКЦИЯ ПОТЕРЬ
# ══════════════════════════════════════════════════════════════════════════════

def test_keras_pinball_matches_the_numpy_definition():
    """
    Обучение минимизирует ровно ту величину, по которой идёт оценка.

    Расхождение между функцией потерь и метрикой не проявилось бы никак: модель
    сходилась бы к другому оптимуму, а отчёт выглядел бы обычно.
    """
    rng = np.random.RandomState(0)
    y = rng.normal(size=(32, 24)).astype(np.float32)
    p = rng.normal(size=(32, 24, 3)).astype(np.float32)

    keras_value = float(tf.reduce_mean(PinballLoss(DEFAULT_QUANTILES)(y, p)))
    manual = float(np.mean([pinball_loss(y, p[:, :, i], q)
                            for i, q in enumerate(DEFAULT_QUANTILES)]))

    assert keras_value == pytest.approx(manual, rel=1e-5)


def test_pinball_loss_survives_serialization():
    """Модель с этой функцией потерь обязана восстанавливаться из файла."""
    loss = PinballLoss([0.1, 0.5, 0.9])
    restored = PinballLoss.from_config(loss.get_config())
    assert restored.quantiles == [0.1, 0.5, 0.9]


def test_broadcast_mistake_would_be_caught():
    """
    Целевая переменная разворачивается по оси уровней, а не сравнивается
    по правилам broadcast.

    При формах (B, H) и (B, H, Q) молчаливый broadcast дал бы величину, которая
    не является pinball ни для одного уровня, но выглядела бы правдоподобно.
    """
    y = np.zeros((4, 24), np.float32)
    p = np.zeros((4, 24, 3), np.float32)
    p[:, :, 0] = -1.0                      # P10 занижен
    p[:, :, 2] = +1.0                      # P90 завышен

    value = float(tf.reduce_mean(PinballLoss([0.1, 0.5, 0.9])(y, p)))
    # Занижение на 1 при q=0.1 стоит 0.1; завышение на 1 при q=0.9 стоит 0.1.
    assert value == pytest.approx((0.1 + 0.0 + 0.1) / 3, rel=1e-5)


# ══════════════════════════════════════════════════════════════════════════════
# ВЕРОЯТНОСТНЫЙ БАЗЛАЙН
# ══════════════════════════════════════════════════════════════════════════════

def test_naive_quantiles_are_ordered_and_cover_reasonably(panel_data):
    """
    Тривиальный базлайн обязан давать осмысленный интервал.

    Без него нельзя утверждать, что обучаемая вероятностная модель что-то
    добавляет: разброс исторических ошибок уже даёт разумное покрытие.
    """
    model = QuantileNaive().fit(panel_data)
    pred = model.predict(make_batch(panel_data, "test"))

    assert pred.shape == panel_data["Y_test"].shape + (3,)
    preds = {q: pred[:, :, i] for i, q in enumerate(DEFAULT_QUANTILES)}

    assert crossing_rate(preds) == 0.0, "квантили остатков не могут пересекаться"
    got = coverage(panel_data["Y_test"], preds[0.1], preds[0.9])
    assert 0.5 < got < 0.98, f"покрытие {got:.2f} неправдоподобно для интервала 80%"


def test_naive_offsets_are_ordered_at_every_horizon_step(panel_data):
    """
    Смещения возрастают по уровням на каждом шаге горизонта.

    Это обязательное свойство: нарушение означало бы, что интервал вывернут
    наизнанку, а форма и число значений при этом остались бы правильными.

    Отдельно зафиксирована измеренная величина: ширина интервала по шагам
    горизонта почти постоянна (отношение крайних около 1.02). Сезонно-наивный
    прогноз повторяет сутки назад, поэтому его ошибка не накапливается с
    удалением горизонта — в отличие от ошибки обучаемой модели, где рост
    заметен. Ожидать здесь роста было бы ошибкой в постановке проверки.
    """
    model = QuantileNaive().fit(panel_data)

    assert model.offsets.shape == (24, 3)
    assert np.all(np.diff(model.offsets, axis=1) >= 0), "уровни не упорядочены"

    width = model.offsets[:, -1] - model.offsets[:, 0]
    assert np.all(width > 0)
    assert width.max() / width.min() < 1.5, (
        "разброс сезонно-наивного прогноза не должен заметно расти по горизонту"
    )


def test_naive_quantiles_come_from_train_only(panel_data):
    """Разброс берётся с обучения: иначе интервал знал бы о тесте."""
    import copy

    model_a = QuantileNaive().fit(panel_data)
    altered = dict(panel_data)
    altered["Y_test"] = np.zeros_like(altered["Y_test"])
    model_b = QuantileNaive().fit(altered)

    assert np.allclose(model_a.offsets, model_b.offsets)


# ══════════════════════════════════════════════════════════════════════════════
# DLINEAR С КВАНТИЛЬНЫМ ВЫХОДОМ
# ══════════════════════════════════════════════════════════════════════════════

def test_quantile_dlinear_shapes_and_finiteness(panel_data):
    nH = len(panel_data["feature_names_hist"])
    nF = len(panel_data["feature_names_future"])
    nS = len(panel_data["static_names"])
    model = build_quantile_dlinear(48, 24, nH, nF, nS)

    batch = make_batch(panel_data, "test")
    out = model.predict([batch["hist"][:16], batch["future"][:16], batch["static"][:16]],
                        verbose=0)

    assert out.shape == (16, 24, 3)
    assert np.isfinite(out).all()


def test_quantile_dlinear_recovers_a_known_distribution():
    """
    Обучение действительно находит квантили, а не три близких числа.

    Проверка на задаче с известным ответом: постоянный вход и шумная цель, где
    истинные квантили вычисляются аналитически. Без такой проверки модель,
    выдающая три почти одинаковых значения, прошла бы все проверки форм.
    """
    rng = np.random.RandomState(0)
    n, horizon = 4096, 4
    X = np.zeros((n, 8, 1), np.float32)
    # Цель центрирована около нуля, как нормированные данные конвейера: модель
    # с нулевым входом ищет ответ только смещениями, и из далёкого начального
    # приближения шагами ограниченного градиента до него просто не дойти.
    Y = rng.normal(0.0, 1.0, size=(n, horizon)).astype(np.float32)

    model = build_quantile_dlinear(8, horizon, 1, kernel_size=3, learning_rate=0.02)
    model.fit(X, Y, epochs=80, batch_size=256, verbose=0)
    pred = model.predict(X[:1], verbose=0)[0, 0]

    expected = [np.quantile(Y, q) for q in DEFAULT_QUANTILES]
    for got, want, q in zip(pred, expected, DEFAULT_QUANTILES):
        assert got == pytest.approx(want, abs=0.2), f"уровень {q}: {got:.2f} против {want:.2f}"

    assert pred[0] < pred[1] < pred[2], "порядок уровней нарушен"


def test_quantile_dlinear_roundtrip(tmp_path, panel_data):
    """Модель сохраняется и восстанавливается с тем же прогнозом."""
    from models.dlinear import SeriesDecomposition

    nH = len(panel_data["feature_names_hist"])
    model = build_quantile_dlinear(48, 24, nH)
    batch = make_batch(panel_data, "test")
    before = model.predict(batch["hist"][:8], verbose=0)

    path = str(tmp_path / "qdlinear.keras")
    model.save(path)
    restored = tf.keras.models.load_model(
        path, custom_objects={"SeriesDecomposition": SeriesDecomposition,
                              "PinballLoss": PinballLoss})
    after = restored.predict(batch["hist"][:8], verbose=0)

    assert np.allclose(before, after, rtol=1e-5, atol=1e-5)
