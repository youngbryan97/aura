from core.utils.readable_text import prose_of, prose_paragraphs


def test_prose_of_keeps_the_paragraphs_and_drops_the_labels():
    page = "\n".join(
        ["Home", "News", "Sport", "Weather"]
        + [
            "The council voted on Tuesday to close the bridge for repairs. "
            "Work starts in March and lasts six weeks.",
            "Buses will run a replacement service from the station.",
        ]
        + ["Terms", "Privacy", "Contact"]
    )

    assert prose_of(page).startswith("The council voted on Tuesday")
    assert "Privacy" not in prose_of(page)


def test_prose_of_returns_the_text_when_nothing_reads_as_prose():
    labels = "Home\nNews\nSport\nWeather"

    assert prose_paragraphs(labels) == []
    assert prose_of(labels) == labels
