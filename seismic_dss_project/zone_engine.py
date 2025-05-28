# zone_engine.py

from experta import (
    KnowledgeEngine, Fact, Field, Rule,
    AS, MATCH, TEST, NOT
)

# ------------------------------------------------------------------
# Региональная экспертная система
# ------------------------------------------------------------------

class RegionStatus(Fact):
    """
    Факт макро-региона:
      region      – имя региона (str)
      gap         – лет с момента последнего значимого события (int)
      prev_rate   – средняя годовая частота в предыдущем периоде (float)
      recent_rate – средняя годовая частота в последнем периоде (float)
      mid_level   – уровень среднесрочного риска (str)
      long_level  – уровень долгосрочного риска (str)
    """
    region      = Field(str,   mandatory=True)
    gap         = Field(int,   mandatory=True)
    prev_rate   = Field(float, mandatory=True)
    recent_rate = Field(float, mandatory=True)
    mid_level   = Field(str,   default=None)
    long_level  = Field(str,   default=None)


class ZoneExpertSystem(KnowledgeEngine):
    @Rule(
        AS.f << RegionStatus(mid_level=None,
                             recent_rate=MATCH.rr,
                             prev_rate=MATCH.pr)
    )
    def assess_mid_term(self, f, rr, pr):
        """Оценка среднесрочного риска по изменению частоты."""
        if pr == 0:
            level = "Высокая" if rr > 0 else "Низкая"
        else:
            ratio = rr / pr
            if ratio >= 1.5 and (rr - pr) > 5:
                level = "Высокая"
            elif rr > 20 or pr > 20:
                level = "Средняя"
            else:
                level = "Низкая"
        self.modify(f, mid_level=level)

    @Rule(
        AS.f << RegionStatus(long_level=None,
                             gap=MATCH.g)
    )
    def assess_long_term(self, f, g):
        """Оценка долгосрочного риска по «разрыву»."""
        if g >= 30:
            level = "Высокая"
        elif g >= 15:
            level = "Средняя"
        else:
            level = "Низкая"
        self.modify(f, long_level=level)

    @Rule(
        AS.f << RegionStatus(region=MATCH.r,
                             mid_level=MATCH.mid,
                             long_level=MATCH.long),
        TEST(lambda mid, long: mid is not None and long is not None)
    )
    def report(self, r, mid, long):
        """Вывод регионального заключения."""
        expected_max = {
            "Западная зона":    18,
            "Центральная зона": 17,
            "Восточная зона":   17,
        }
        if mid == "Высокая":
            max_cls = expected_max.get(r, 17)
            rng = f"{max_cls-2}–{max_cls}"
        elif mid == "Средняя":
            rng = "10–15"
        else:
            rng = "<10"
        print(f"Регион: {r} | Среднесрочный риск: {mid}, "
              f"Долгосрочный риск: {long}. Ожидаемый класс: {rng}.")


# ------------------------------------------------------------------
# Экспертная система для «горячих точек»
# ------------------------------------------------------------------

class CellStatus(Fact):
    """
    Факт ячейки сетки:
      lon  – долгота центра (float)
      lat  – широта центра (float)
      prob – прогнозная вероятность события (float)
    """
    lon  = Field(float, mandatory=True)
    lat  = Field(float, mandatory=True)
    prob = Field(float, mandatory=True)


class _Done(Fact):
    """Метка, что отчёт по топ-10 уже выведен."""


class CoordinateExpertSystem(KnowledgeEngine):

    @Rule(
        NOT(_Done()),      # срабатывает один раз
        salience=-100      # после всех других правил
    )
    def report_top(self):
        """Вывод 10 ячеек с наивысшей вероятностью."""
        # Собираем все факты ячеек
        cells = [
            fact for fact in self.facts.values()
            if isinstance(fact, CellStatus)
        ]
        # Сортируем по prob и берём топ-10
        top10 = sorted(cells, key=lambda c: c["prob"], reverse=True)[:10]

        print("\n=== Top 10 «горячих точек» ===")
        for c in top10:
            p = c["prob"]
            # классификация уровня риска на лету
            if p > 0.005:
                lvl = "Высокий"
            elif p > 0.001:
                lvl = "Средний"
            else:
                lvl = "Низкий"
            print(f"[{lvl}] λ≈{p:.2%}  ({c['lat']:.2f}°N, {c['lon']:.2f}°E)")

        # Обозначаем, что отчёт выполнен
        self.declare(_Done())
