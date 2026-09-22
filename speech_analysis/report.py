"""Текстовый отчёт. Только форматирование, никаких вычислений."""


def mmss(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def render(data: dict, cfg: dict) -> str:
    show = cfg["show"]
    people, out = data["people"], []
    add = out.append

    add("=" * 62)
    add(f"РЕЧЕВЫЕ МЕТРИКИ: {data['source']}")
    add("=" * 62)
    add(f"Длительность: {mmss(data['span'])}, реплик: {data['turns']}, "
        f"говорящих: {len(people)}")
    add("")

    total = sum(p["speaking"] for p in people.values()) or 1
    for spk, p in sorted(people.items(), key=lambda kv: -kv[1]["speaking"]):
        add(f"── {spk}")
        if show["balance"]:
            add(f"   Времени в эфире: {mmss(p['speaking'])} "
                f"({p['speaking'] / total * 100:.0f}%)")
            add(f"   Реплик: {len(p['turns'])}, слов: {p['words']}")
            add(f"   Средняя реплика: {p['turn_mean']:.0f} с, "
                f"самая длинная: {p['turn_max']:.0f} с")
        if show["rate"]:
            add(f"   Темп речи: {p['rate']:.0f} слов/мин "
                f"(разброс ±{p['rate_spread']:.0f})")
        if show["parasites"]:
            share = p["parasites"] / p["words"] * 100 if p["words"] else 0
            add(f"   Слова-паразиты: {share:.1f}% речи ({p['parasites']} шт.)")
        if show["questions"]:
            add(f"   Вопросов задано: {p['questions']}")
        if show["transitions"] and data["overlaps"].get(spk):
            add(f"   Вступал поверх собеседника: {data['overlaps'][spk]} раз")
        add("")

    if show["inner"]:
        add("── Внутри реплик")
        if data["inner"] is None:
            add(f"   Нет файла с пословными таймкодами — паузы и запинки")
            add("   внутри реплики без него не считаются.")
        else:
            for spk, d in sorted(data["inner"].items(), key=lambda kv: -kv[1]["words"]):
                if not d["words"]:
                    continue
                short = [g for g in d["pauses"] if g >= cfg["pause_short"]]
                long_ = [g for g in d["pauses"] if g >= cfg["pause_long"]]
                add(f"   {spk}:")
                line = f"      Пауз дольше {cfg['pause_short']:g} с: {len(short)}"
                if long_:
                    line += f" (дольше {cfg['pause_long']:g} с: {len(long_)})"
                if d["pauses"]:
                    line += f", самая длинная {max(d['pauses']):.1f} с"
                add(line)
                if d["words"]:
                    add(f"      На 100 слов пауз: {len(short) / d['words'] * 100:.1f}")
                total_h = sum(d["hesitations"].values())
                if total_h:
                    parts = ", ".join(f"{k} — {v}" for k, v in
                                      sorted(d["hesitations"].items(), key=lambda kv: -kv[1]))
                    add(f"      Запинок: {total_h} ({parts})")
                    add(f"      На 100 слов: {total_h / d['words'] * 100:.1f}")
                else:
                    add("      Запинок не найдено — возможно, система их вычищает.")
        add("")

    if show["transitions"]:
        add("── Между репликами")
        pauses = data["pauses"]
        if pauses:
            import statistics as st
            long_ = [g for g in pauses if g >= 2]
            add(f"   Смен говорящего: {len(pauses)}, медиана паузы: "
                f"{st.median(pauses):.1f} с")
            add(f"   Пауз дольше 2 с: {len(long_)}, "
                f"самая длинная {max(pauses):.0f} с")
        else:
            add("   Смен говорящего не зафиксировано.")
        add("")

    add("Метрики объективны, но зависят от данных. Время реплик округлено до")
    add("секунды: темп на коротких репликах не считается. Запинки видны только")
    add("там, где система их сохраняет — Whisper причёсывает речь, ElevenLabs")
    add("и Deepgram оставляют, поэтому сравнивать их счёт между системами")
    add("бессмысленно.")
    add("=" * 62)
    return "\n".join(out)
