import re


def string_cleaner(text: str) -> str:
    """
    Clean a string of text. Remove possible spaces before punctuation and format for proper capitalization.

    Parameters
    ----------
    text: str
        A string of text
    Returns
    -------
    str
        Formatted string.
    """
    text = re.sub(r'\s([?.,%!"](?:\s|$))', r"\1", text)

    text = ". ".join(map(lambda s: s.strip().capitalize(), text.split(".")))
    text = "? ".join(map(lambda s: s.strip().capitalize(), text.split("?")))
    text = "! ".join(map(lambda s: s.strip().capitalize(), text.split("!")))
    text = " ' ".join(map(lambda s: s.strip(), text.split("'")))
    text = text.replace(" ' ", "'")
    return text
