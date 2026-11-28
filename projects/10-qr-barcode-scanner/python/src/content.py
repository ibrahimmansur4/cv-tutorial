"""Classify what a decoded payload actually *is*.

A scanner that only prints the raw string makes the user do the parsing. The
payload almost always announces its own type, because QR codes carry a handful
of conventional formats: a URI scheme, or one of the pseudo-schemes that phone
cameras act on -- `WIFI:`, `BEGIN:VCARD`, `mailto:`, `tel:`, `geo:`, `SMSTO:`.

None of this is part of the QR standard. The standard stops at "here are your
bytes"; everything above it is convention that decoders agreed on. That is
worth knowing, because it means a payload can be perfectly valid and still be
unrecognisable here -- in which case it is plain text, which is the honest
answer rather than an error.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class ContentKind(str, Enum):
    """The recognised payload conventions, plus a fallback."""

    URL = "url"
    WIFI = "wifi"
    CONTACT = "contact"
    EMAIL = "email"
    PHONE = "phone"
    SMS = "sms"
    GEO = "geo"
    CALENDAR = "calendar"
    PRODUCT = "product"
    TEXT = "text"


@dataclass(frozen=True, slots=True)
class Content:
    """A decoded payload and what it was recognised as."""

    kind: ContentKind
    raw: str
    summary: str

    @property
    def is_url(self) -> bool:
        return self.kind is ContentKind.URL


# A scheme must look like a real URI scheme: a letter, then letters, digits,
# "+", "-" or "." before the colon. Testing for a bare ":" would classify
# "Note: buy milk" as a URL.
_URI_SCHEME = re.compile(r"^(?P<scheme>[A-Za-z][A-Za-z0-9+.\-]*):")

# Non-hierarchical schemes a scanner is plausibly handed, which have no "//"
# after the colon. Without a list like this, the only way to tell "bitcoin:1A.."
# from "Reminder:call Sam" is that one of them is a registered scheme.
_OPAQUE_SCHEMES = frozenset(
    {"bitcoin", "ethereum", "upi", "otpauth", "market", "magnet", "spotify", "matmsg"}
)

ESCAPE = chr(92)  # A single backslash, named to keep the loop below readable.


def _clip(text: str, limit: int = 72) -> str:
    """Shorten for a one-line display, marking that it was shortened."""
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1] + "…"


def parse_wifi(payload: str) -> dict[str, str]:
    """Pull the fields out of a `WIFI:T:WPA;S:name;P:secret;;` payload.

    Written as an explicit scan rather than a regex because the format allows
    backslash escapes, so a password of `a;b` is legal and must not split the
    field. A regex that handles that correctly is harder to read than the loop.
    """
    body = payload[5:] if payload[:5].upper() == "WIFI:" else payload
    fields: dict[str, str] = {}
    key = ""
    value: list[str] = []
    expecting_key = True
    index = 0
    while index < len(body):
        character = body[index]
        if character == ESCAPE and index + 1 < len(body):
            # An escaped character is data, whatever it happens to be.
            value.append(body[index + 1])
            index += 2
            continue
        if expecting_key and character == ":":
            key = "".join(value).upper()
            value = []
            expecting_key = False
        elif character == ";":
            if key:
                fields[key] = "".join(value)
            key = ""
            value = []
            expecting_key = True
        else:
            value.append(character)
        index += 1
    if key and not expecting_key:
        fields[key] = "".join(value)
    return fields


def _wifi_summary(payload: str) -> str:
    fields = parse_wifi(payload)
    ssid = fields.get("S", "?")
    security = fields.get("T", "").upper()
    security = "open" if security in {"NOPASS", ""} else security
    hidden = " (hidden)" if fields.get("H", "").lower() == "true" else ""
    return f"Wi-Fi {ssid} [{security}]{hidden}"


def _contact_summary(payload: str) -> str:
    for line in payload.splitlines():
        if line[:3].upper() == "FN:":
            return f"Contact {_clip(line[3:], 48)}"
        if line[:2].upper() == "N:":
            parts = [piece for piece in line[2:].split(";") if piece]
            if parts:
                # vCard "N:" is family-name-first; show it the way people read it.
                return f"Contact {_clip(' '.join(reversed(parts[:2])), 48)}"
    return "Contact card"


def _calendar_summary(payload: str) -> str:
    for line in payload.splitlines():
        if line[:8].upper() == "SUMMARY:":
            return f"Event {_clip(line[8:], 48)}"
    return "Calendar event"


def classify(payload: str, *, symbology: str = "") -> Content:
    """Recognise a decoded payload.

    `symbology` lets a 1D barcode be identified as a product code without
    guessing from the digits: an EAN-13 is a product number by definition,
    while the same thirteen digits in a QR code are just digits.
    """
    text = payload.strip()
    if not text:
        return Content(ContentKind.TEXT, payload, "(empty)")

    if symbology and not symbology.upper().startswith("QR"):
        return Content(ContentKind.PRODUCT, payload, f"{symbology} {_clip(text, 40)}")

    upper = text.upper()
    if upper.startswith("WIFI:"):
        return Content(ContentKind.WIFI, payload, _wifi_summary(text))
    if upper.startswith("BEGIN:VCARD") or upper.startswith("MECARD:"):
        return Content(ContentKind.CONTACT, payload, _contact_summary(text))
    if upper.startswith("BEGIN:VEVENT") or upper.startswith("BEGIN:VCALENDAR"):
        return Content(ContentKind.CALENDAR, payload, _calendar_summary(text))

    match = _URI_SCHEME.match(text)
    if match:
        scheme = match.group("scheme").lower()
        rest = text[match.end() :]
        if scheme in {"http", "https"}:
            return Content(ContentKind.URL, payload, _clip(text))
        if scheme == "mailto":
            return Content(ContentKind.EMAIL, payload, f"Email {_clip(rest, 48)}")
        if scheme in {"tel", "facetime"}:
            return Content(ContentKind.PHONE, payload, f"Phone {_clip(rest, 32)}")
        if scheme in {"sms", "smsto"}:
            return Content(ContentKind.SMS, payload, f"SMS {_clip(rest, 48)}")
        if scheme == "geo":
            return Content(ContentKind.GEO, payload, f"Location {_clip(rest, 40)}")
        # An unknown scheme is only a link if it looks like one. Prose such as
        # "Note: buy milk" matches the scheme pattern too, so require either the
        # "//" of a hierarchical URI or a scheme known to go without it. A space
        # straight after the colon rules out a URI either way.
        if rest.startswith("//") or (scheme in _OPAQUE_SCHEMES and rest[:1].strip()):
            return Content(ContentKind.URL, payload, _clip(text))

    return Content(ContentKind.TEXT, payload, _clip(text))
