"""Tests for the QR and barcode scanner.

Every fixture is rendered by the project itself, so the suite needs no sample
images and every assertion has a known ground-truth payload to check against.
"""

from __future__ import annotations

import contextlib
import csv
import io
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from src import synth
from src.app import (
    AppConfig,
    FpsMeter,
    build_parser,
    main,
    parse_size,
    resolve_config,
    scale_factor,
    zoom_fraction,
)
from src.content import ContentKind, classify, parse_wifi
from src.detect import (
    BarcodeReader,
    Detection,
    MultiReader,
    QRReader,
    ScanOptions,
    _as_quads,
    _merge,
    build_reader,
    retail_key,
    same_payload,
)
from src.errors import ConfigurationError, HistoryError, MediaError
from src.history import CSV_FIELDS, ScanHistory, ScanLog
from src.overlay import HudState, annotate, draw_label
from src.symbols import (
    EAN13_MODULES,
    ean13_check_digit,
    ean13_modules,
    normalise_ean13,
    on_canvas,
    render_barcode,
    render_ean13,
    render_qr,
)

FRAME = (1280, 720)


@contextlib.contextmanager
def quiet():
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        yield buffer


def qr_frame(payload: str = "https://example.com/scan", **kwargs: object) -> synth.Scene:
    code = render_qr(payload, **kwargs)  # type: ignore[arg-type]
    return synth.place(code, width=FRAME[0], height=FRAME[1])


def ean_frame(digits: str = "400638133393", module: int = 4) -> synth.Scene:
    code = render_barcode(digits, module_width=module, height=69 * module)
    return synth.place(code, width=FRAME[0], height=FRAME[1])


def quad(x: float, y: float, size: float = 10.0) -> np.ndarray:
    return np.array(
        [[x, y], [x + size, y], [x + size, y + size], [x, y + size]], dtype=np.float32
    )


# --------------------------------------------------------------------------
# EAN-13 encoding
# --------------------------------------------------------------------------


class CheckDigitTests(unittest.TestCase):
    def test_known_check_digits(self):
        """Published examples, so the weighting cannot be subtly wrong."""
        for body, expected in [
            ("400638133393", 1),
            ("978014300723", 4),
            ("590123412345", 7),
            ("000000000000", 0),
        ]:
            self.assertEqual(ean13_check_digit(body), expected, body)

    def test_check_digit_rejects_wrong_length(self):
        for bad in ["12345", "1234567890123"]:
            with self.assertRaises(ConfigurationError):
                ean13_check_digit(bad)

    def test_check_digit_rejects_non_digits(self):
        with self.assertRaises(ConfigurationError):
            ean13_check_digit("40063813339x")

    def test_normalise_appends_check_digit(self):
        self.assertEqual(normalise_ean13("400638133393"), "4006381333931")

    def test_normalise_accepts_valid_thirteen(self):
        self.assertEqual(normalise_ean13("4006381333931"), "4006381333931")

    def test_normalise_rejects_bad_check_digit(self):
        """A wrong check digit is an error, not something to silently repair."""
        with self.assertRaises(ConfigurationError) as caught:
            normalise_ean13("4006381333932")
        self.assertIn("check digit", str(caught.exception))

    def test_normalise_rejects_odd_length(self):
        with self.assertRaises(ConfigurationError):
            normalise_ean13("4006381")

    def test_normalise_ignores_whitespace(self):
        self.assertEqual(normalise_ean13(" 400638 133393 "), "4006381333931")


class ModulePatternTests(unittest.TestCase):
    def test_pattern_is_ninety_five_modules(self):
        self.assertEqual(len(ean13_modules("4006381333931")), EAN13_MODULES)

    def test_guard_patterns_are_in_place(self):
        bits = "".join(str(bit) for bit in ean13_modules("4006381333931"))
        self.assertEqual(bits[0:3], "101")
        self.assertEqual(bits[45:50], "01010")
        self.assertEqual(bits[92:95], "101")

    def test_first_digit_changes_left_parity_not_the_digits(self):
        """The leading digit is carried by the choice of alphabet, nothing else."""
        first = ean13_modules(normalise_ean13("100000000000"))
        second = ean13_modules(normalise_ean13("200000000000"))
        self.assertFalse(np.array_equal(first[3:45], second[3:45]))

    def test_pattern_is_binary(self):
        self.assertEqual(set(np.unique(ean13_modules("4006381333931"))), {0, 1})


class RenderTests(unittest.TestCase):
    def test_barcode_image_has_expected_width(self):
        image = render_ean13("4006381333931", module_width=3, quiet_zone=11, height=120)
        self.assertEqual(image.shape, (120, (EAN13_MODULES + 22) * 3))

    def test_barcode_is_two_level(self):
        image = render_ean13("4006381333931")
        self.assertEqual(set(np.unique(image)), {0, 255})

    def test_barcode_rejects_tiny_geometry(self):
        with self.assertRaises(ConfigurationError):
            render_ean13("4006381333931", module_width=0)
        with self.assertRaises(ConfigurationError):
            render_ean13("4006381333931", height=5)

    def test_guard_bars_run_lower_than_data_bars(self):
        """Guards are drawn long, as on a real label."""
        image = render_ean13("4006381333931", module_width=4, height=100, guard_extension=10)
        quiet_pixels = 11 * 4
        self.assertEqual(image[-1, quiet_pixels], 0)  # left guard reaches the bottom
        column = quiet_pixels + 3 * 4 + 2  # inside the first data digit
        self.assertEqual(image[-1, column], 255)

    def test_qr_quiet_zone_is_measured_not_assumed(self):
        """OpenCV adds a margin of its own; the recorded one includes it."""
        code = render_qr("hello", module_size=8, border=4)
        self.assertGreaterEqual(code.quiet[0], 32)
        left, top, width, height = code.inner
        self.assertEqual((left, top), code.quiet)
        self.assertEqual(width, code.size[0] - 2 * code.quiet[0])

    def test_qr_inner_box_starts_at_a_dark_module(self):
        """The measured margin must land exactly on the finder pattern."""
        code = render_qr("hello", module_size=8, border=4)
        left, top, _, _ = code.inner
        self.assertEqual(int(code.image[top, left]), 0)
        self.assertEqual(int(code.image[top - 1, left - 1]), 255)

    def test_qr_rejects_empty_payload(self):
        with self.assertRaises(ConfigurationError):
            render_qr("")

    def test_qr_rejects_unknown_correction_level(self):
        with self.assertRaises(ConfigurationError):
            render_qr("hello", correction="Z")

    def test_higher_correction_needs_a_bigger_symbol(self):
        """More parity means more modules for the same payload."""
        low = render_qr("https://example.com/scan", correction="L")
        high = render_qr("https://example.com/scan", correction="H")
        self.assertGreater(high.size[0], low.size[0])

    def test_barcode_quiet_zone_is_horizontal_only(self):
        code = render_barcode("400638133393", module_width=4)
        self.assertEqual(code.quiet[1], 0)
        self.assertGreater(code.quiet[0], 0)


# --------------------------------------------------------------------------
# Round trip: our encoder against OpenCV's decoder
# --------------------------------------------------------------------------


class RoundTripTests(unittest.TestCase):
    """The strongest correctness check available: encode, then decode back."""

    def test_qr_round_trips(self):
        for payload in [
            "hello",
            "https://example.com/scan",
            "WIFI:T:WPA;S:Net;P:secret;;",
            "tel:+441234567890",
        ]:
            scene = qr_frame(payload)
            found = build_reader(barcode=False).read(scene.image)
            self.assertIn(payload, [item.payload for item in found], payload)

    def test_ean13_round_trips(self):
        """Our own bar pattern, read back by OpenCV."""
        for digits in ["400638133393", "978014300723", "590123412345", "888888888888"]:
            scene = ean_frame(digits)
            found = build_reader(qr=False).read(scene.image)
            keys = {retail_key(item.payload, item.symbology) for item in found}
            self.assertIn(normalise_ean13(digits), keys, digits)

    def test_leading_zero_ean13_is_read_as_upc_a(self):
        """A leading-zero EAN-13 *is* a UPC-A, and is reported as one."""
        scene = ean_frame("012345678905")
        found = build_reader(qr=False).read(scene.image)
        self.assertTrue(found)
        self.assertEqual(found[0].symbology, "UPC_A")
        self.assertEqual(
            retail_key(found[0].payload, found[0].symbology), "0123456789050"
        )

    def test_qr_and_barcode_in_one_frame(self):
        codes = [render_qr("https://example.com/scan"), render_barcode("400638133393")]
        scene = synth.grid(codes, width=1800, height=900, columns=2)
        found = build_reader().read(scene.image)
        payloads = {retail_key(item.payload, item.symbology) for item in found}
        self.assertIn("https://example.com/scan", payloads)
        self.assertIn("4006381333931", payloads)


# --------------------------------------------------------------------------
# Detection plumbing
# --------------------------------------------------------------------------


class QuadTests(unittest.TestCase):
    def test_none_gives_no_quads(self):
        self.assertEqual(_as_quads(None), [])

    def test_empty_array_gives_no_quads(self):
        self.assertEqual(_as_quads(np.zeros((0, 4, 2), dtype=np.float32)), [])

    def test_flat_eight_values_become_one_quad(self):
        quads = _as_quads(np.arange(8, dtype=np.float32))
        self.assertEqual(len(quads), 1)
        self.assertEqual(quads[0].shape, (4, 2))

    def test_several_quads_are_kept_separate(self):
        self.assertEqual(len(_as_quads(np.zeros((3, 4, 2), dtype=np.float32))), 3)


class DetectionTests(unittest.TestCase):
    def test_area_of_a_unit_square(self):
        self.assertAlmostEqual(Detection("x", "QR", quad(0, 0, 10)).area, 100.0)

    def test_centre_of_a_square(self):
        self.assertEqual(Detection("x", "QR", quad(0, 0, 10)).centre, (5.0, 5.0))

    def test_bounds_are_integral_and_inclusive(self):
        self.assertEqual(Detection("x", "QR", quad(3.2, 4.8, 10)).bounds, (3, 4, 11, 11))

    def test_empty_payload_is_not_decoded(self):
        self.assertFalse(Detection("", "QR", quad(0, 0)).decoded)

    def test_undecoded_label_says_so(self):
        self.assertIn("not decoded", Detection("", "QR", quad(0, 0)).label)

    def test_with_content_classifies_the_payload(self):
        detection = Detection("https://example.com", "QR", quad(0, 0)).with_content()
        self.assertIsNotNone(detection.content)
        self.assertEqual(detection.content.kind, ContentKind.URL)

    def test_with_content_leaves_undecoded_alone(self):
        detection = Detection("", "QR", quad(0, 0)).with_content()
        self.assertIsNone(detection.content)

    def test_scaled_maps_corners_back(self):
        detection = Detection("x", "QR", quad(10, 20, 10)).scaled(2.0)
        self.assertAlmostEqual(float(detection.corners[0][0]), 5.0)
        self.assertAlmostEqual(float(detection.corners[0][1]), 10.0)

    def test_scaled_by_one_is_a_no_op(self):
        original = Detection("x", "QR", quad(10, 20))
        self.assertIs(original.scaled(1.0), original)


class RetailKeyTests(unittest.TestCase):
    def test_upc_a_and_ean13_agree(self):
        self.assertEqual(retail_key("123456789050", "UPC_A"), retail_key("0123456789050", "EAN_13"))

    def test_qr_payloads_are_untouched(self):
        self.assertEqual(retail_key("0012", "QR"), "0012")

    def test_non_numeric_barcode_payload_is_untouched(self):
        self.assertEqual(retail_key("ABC", "EAN_13"), "ABC")

    def test_same_payload_across_symbologies(self):
        left = Detection("123456789050", "UPC_A", quad(0, 0))
        right = Detection("0123456789050", "EAN_13", quad(0, 0))
        self.assertTrue(same_payload(left, right))


class MergeTests(unittest.TestCase):
    def test_coincident_duplicates_collapse(self):
        first = Detection("same", "QR", quad(0, 0, 40))
        second = Detection("same", "QR", quad(1, 1, 40))
        self.assertEqual(len(_merge([first, second])), 1)

    def test_decoded_wins_over_undecoded(self):
        decoded = Detection("payload", "QR", quad(0, 0, 40))
        blank = Detection("", "QR", quad(1, 1, 40))
        merged = _merge([blank, decoded])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].payload, "payload")

    def test_distant_codes_are_both_kept(self):
        self.assertEqual(
            len(_merge([Detection("a", "QR", quad(0, 0, 40)), Detection("b", "QR", quad(500, 500, 40))])),
            2,
        )

    def test_different_payloads_in_one_place_are_both_kept(self):
        """Two stacked codes are two codes, not a duplicate."""
        merged = _merge(
            [Detection("a", "QR", quad(0, 0, 40)), Detection("b", "QR", quad(2, 2, 40))]
        )
        self.assertEqual(len(merged), 2)

    def test_upc_and_ean_readings_of_one_symbol_collapse(self):
        merged = _merge(
            [
                Detection("123456789050", "UPC_A", quad(0, 0, 40)),
                Detection("0123456789050", "EAN_13", quad(1, 1, 40)),
            ]
        )
        self.assertEqual(len(merged), 1)


class ReaderTests(unittest.TestCase):
    def test_build_reader_rejects_scanning_for_nothing(self):
        with self.assertRaises(ConfigurationError):
            build_reader(qr=False, barcode=False)

    def test_unknown_qr_detector_is_rejected(self):
        with self.assertRaises(ConfigurationError):
            QRReader(detector="magic")

    def test_zoom_ladder_is_used_by_default(self):
        self.assertEqual(BarcodeReader().ladder, BarcodeReader.ZOOM_LADDER)

    def test_explicit_zoom_replaces_the_ladder(self):
        self.assertEqual(BarcodeReader(zoom_out=0.3).ladder, (0.3,))

    def test_zoom_of_one_disables_the_retry(self):
        self.assertEqual(BarcodeReader(zoom_out=1.0).ladder, ())

    def test_invalid_zoom_is_rejected(self):
        for bad in (0.0, -1.0, 1.5):
            with self.assertRaises(ConfigurationError):
                BarcodeReader(zoom_out=bad)

    def test_zoom_retry_rescues_a_frame_filling_barcode(self):
        """The finding this project's barcode support rests on."""
        code = render_barcode("400638133393", module_width=6, height=69 * 6)
        tight = on_canvas(code, margin=12, colour=False)
        self.assertEqual(MultiReader([BarcodeReader(zoom_out=1.0)]).read(tight), [])
        found = MultiReader([BarcodeReader()]).read(tight)
        self.assertEqual([item.payload for item in found], [code.payload])

    def test_zoom_retry_reports_corners_in_frame_coordinates(self):
        """A rescued detection must not be left in canvas coordinates."""
        code = render_barcode("400638133393", module_width=6, height=69 * 6)
        tight = on_canvas(code, margin=12, colour=False)
        found = MultiReader([BarcodeReader()]).read(tight)
        self.assertTrue(found)
        height, width = tight.shape[:2]
        corners = found[0].corners
        self.assertTrue((corners[:, 0] >= -2).all() and (corners[:, 0] <= width + 2).all())
        self.assertTrue((corners[:, 1] >= -2).all() and (corners[:, 1] <= height + 2).all())

    def test_retry_box_agrees_with_the_native_detection(self):
        """The rescued corners must land where an unaided detection puts them.

        Detecting the same symbol twice -- once small in a large frame, where no
        retry is needed, and once filling a tight crop, where it is -- and
        comparing both boxes *relative to the symbol* is what proves the offset
        is undone correctly. Checking only that the corners fall inside the
        frame would pass even if they were off by the padding.
        """
        code = render_barcode("400638133393", module_width=6, height=69 * 6)
        wide = synth.place(code, width=2400, height=1400)
        native = MultiReader([BarcodeReader()]).read(wide.image)
        self.assertTrue(native)
        left, top, right, bottom = wide.placements[0].symbol_box
        native_offset = (
            native[0].corners[:, 0].min() - left,
            native[0].corners[:, 1].min() - top,
        )

        tight = on_canvas(code, margin=12, colour=False)
        rescued = MultiReader([BarcodeReader()]).read(tight)
        self.assertTrue(rescued)
        tight_left = 12 + code.quiet[0]
        tight_top = 12 + code.quiet[1]
        rescued_offset = (
            rescued[0].corners[:, 0].min() - tight_left,
            rescued[0].corners[:, 1].min() - tight_top,
        )
        self.assertAlmostEqual(rescued_offset[0], native_offset[0], delta=12.0)
        self.assertAlmostEqual(rescued_offset[1], native_offset[1], delta=12.0)

    def test_empty_frame_yields_nothing(self):
        self.assertEqual(build_reader().read(np.zeros((0, 0), dtype=np.uint8)), [])

    def test_blank_frame_yields_nothing(self):
        blank = np.full((480, 640, 3), 235, dtype=np.uint8)
        self.assertEqual(build_reader().read(blank), [])

    def test_min_area_filters_small_codes(self):
        scene = qr_frame()
        generous = MultiReader([QRReader()], ScanOptions(min_area=1.0)).read(scene.image)
        self.assertTrue(generous)
        strict = MultiReader([QRReader()], ScanOptions(min_area=1e9)).read(scene.image)
        self.assertEqual(strict, [])

    def test_upscale_still_reports_original_coordinates(self):
        """Detecting on an enlarged copy must not move the reported corners."""
        scene = qr_frame()
        plain = MultiReader([QRReader()]).read(scene.image)
        scaled = MultiReader([QRReader()], ScanOptions(upscale=2.0)).read(scene.image)
        self.assertTrue(plain and scaled)
        self.assertLess(
            float(np.linalg.norm(np.array(plain[0].centre) - np.array(scaled[0].centre))), 8.0
        )

    def test_single_code_fallback_rescues_the_classic_detector(self):
        """The multi path drops codes the single path reads."""
        scene = qr_frame("https://example.com/scan")
        reader = QRReader(detector="classic")
        gray = cv2.cvtColor(scene.image, cv2.COLOR_BGR2GRAY)
        self.assertEqual(reader._read_multi(gray), [])
        self.assertTrue(reader.read(gray)[0].decoded)

    def test_detections_carry_classified_content(self):
        found = build_reader(barcode=False).read(qr_frame("https://example.com/scan").image)
        self.assertEqual(found[0].content.kind, ContentKind.URL)

    def test_barcode_content_is_a_product(self):
        found = build_reader(qr=False).read(ean_frame().image)
        self.assertEqual(found[0].content.kind, ContentKind.PRODUCT)


# --------------------------------------------------------------------------
# Content classification
# --------------------------------------------------------------------------


class ClassifyTests(unittest.TestCase):
    def test_recognised_kinds(self):
        cases = [
            ("https://example.com", ContentKind.URL),
            ("http://example.com/a?b=c", ContentKind.URL),
            ("WIFI:T:WPA;S:Net;P:secret;;", ContentKind.WIFI),
            ("BEGIN:VCARD\nFN:Ada\nEND:VCARD", ContentKind.CONTACT),
            ("mailto:someone@example.com", ContentKind.EMAIL),
            ("tel:+441234567890", ContentKind.PHONE),
            ("SMSTO:+44123:hello", ContentKind.SMS),
            ("geo:51.5,-0.1", ContentKind.GEO),
            ("BEGIN:VEVENT\nSUMMARY:Standup\nEND:VEVENT", ContentKind.CALENDAR),
            ("just some words", ContentKind.TEXT),
        ]
        for payload, expected in cases:
            self.assertEqual(classify(payload).kind, expected, payload)

    def test_prose_with_a_colon_is_not_a_url(self):
        """The trap a naive scheme test falls into."""
        for payload in ["Note: buy milk", "TODO: ship it", "Warning: hot"]:
            self.assertEqual(classify(payload).kind, ContentKind.TEXT, payload)

    def test_opaque_scheme_is_a_link(self):
        self.assertEqual(classify("bitcoin:1A1zP1eP").kind, ContentKind.URL)

    def test_unknown_hierarchical_scheme_is_a_link(self):
        self.assertEqual(classify("myapp://open/thing").kind, ContentKind.URL)

    def test_barcode_symbology_forces_product(self):
        content = classify("4006381333931", symbology="EAN_13")
        self.assertEqual(content.kind, ContentKind.PRODUCT)

    def test_same_digits_in_a_qr_are_not_a_product(self):
        self.assertEqual(classify("4006381333931", symbology="QR").kind, ContentKind.TEXT)

    def test_empty_payload(self):
        self.assertEqual(classify("   ").summary, "(empty)")

    def test_long_payload_is_clipped(self):
        summary = classify("x" * 500).summary
        self.assertLessEqual(len(summary), 72)
        self.assertTrue(summary.endswith("…"))

    def test_wifi_summary_names_the_network(self):
        self.assertIn("Workshop", classify("WIFI:T:WPA;S:Workshop;P:p;;").summary)

    def test_open_network_is_described_as_open(self):
        self.assertIn("open", classify("WIFI:S:Guest;;").summary)

    def test_contact_summary_uses_the_formatted_name(self):
        self.assertIn("Ada Lovelace", classify("BEGIN:VCARD\nFN:Ada Lovelace\nEND:VCARD").summary)

    def test_contact_summary_falls_back_to_structured_name(self):
        self.assertIn("Ada", classify("BEGIN:VCARD\nN:Lovelace;Ada\nEND:VCARD").summary)


class WifiParseTests(unittest.TestCase):
    def test_fields_are_extracted(self):
        fields = parse_wifi("WIFI:T:WPA;S:MyNet;P:secret;;")
        self.assertEqual(fields["T"], "WPA")
        self.assertEqual(fields["S"], "MyNet")
        self.assertEqual(fields["P"], "secret")

    def test_escaped_semicolon_stays_in_the_password(self):
        """An escaped separator is data; splitting on it corrupts the password."""
        fields = parse_wifi("WIFI:S:Net;P:pa" + chr(92) + ";ss;;")
        self.assertEqual(fields["P"], "pa;ss")

    def test_escaped_backslash_survives(self):
        fields = parse_wifi("WIFI:S:Net;P:a" + chr(92) + chr(92) + "b;;")
        self.assertEqual(fields["P"], "a" + chr(92) + "b")

    def test_hidden_flag_is_reported(self):
        self.assertIn("hidden", classify("WIFI:S:Net;H:true;;").summary)

    def test_missing_trailing_semicolon_still_parses(self):
        self.assertEqual(parse_wifi("WIFI:S:Net")["S"], "Net")


# --------------------------------------------------------------------------
# History and the CSV log
# --------------------------------------------------------------------------


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def detection(payload: str = "p", symbology: str = "QR") -> Detection:
    return Detection(payload, symbology, quad(0, 0, 20)).with_content()


class HistoryTests(unittest.TestCase):
    def test_first_scan_is_accepted(self):
        history = ScanHistory(cooldown=3.0, clock=FakeClock())
        self.assertIsNotNone(history.offer(detection()))
        self.assertEqual(len(history), 1)

    def test_repeat_inside_the_cooldown_is_suppressed(self):
        clock = FakeClock()
        history = ScanHistory(cooldown=3.0, clock=clock)
        history.offer(detection())
        clock.advance(1.0)
        self.assertIsNone(history.offer(detection()))
        self.assertEqual(len(history), 1)
        self.assertEqual(history.suppressed, 1)

    def test_repeat_after_the_cooldown_is_accepted(self):
        """A deliberate re-scan must still work; a seen-set would block it."""
        clock = FakeClock()
        history = ScanHistory(cooldown=3.0, clock=clock)
        history.offer(detection())
        clock.advance(4.0)
        self.assertIsNotNone(history.offer(detection()))
        self.assertEqual(len(history), 2)

    def test_holding_a_code_in_view_keeps_it_suppressed(self):
        """The window refreshes, so a held code does not resurface every cooldown."""
        clock = FakeClock()
        history = ScanHistory(cooldown=3.0, clock=clock)
        history.offer(detection())
        for _ in range(20):
            clock.advance(0.5)
            history.offer(detection())
        self.assertEqual(len(history), 1)

    def test_different_payloads_are_independent(self):
        history = ScanHistory(cooldown=100.0, clock=FakeClock())
        history.offer(detection("a"))
        history.offer(detection("b"))
        self.assertEqual(len(history), 2)

    def test_same_digits_in_different_symbologies_are_separate_scans(self):
        history = ScanHistory(cooldown=100.0, clock=FakeClock())
        history.offer(detection("123", "QR"))
        history.offer(detection("123", "EAN_13"))
        self.assertEqual(len(history), 2)

    def test_undecoded_detections_are_never_recorded(self):
        history = ScanHistory(cooldown=0.0)
        self.assertIsNone(history.offer(Detection("", "QR", quad(0, 0))))
        self.assertEqual(len(history), 0)

    def test_zero_cooldown_accepts_everything(self):
        history = ScanHistory(cooldown=0.0, clock=FakeClock())
        for _ in range(3):
            history.offer(detection())
        self.assertEqual(len(history), 3)

    def test_negative_cooldown_is_rejected(self):
        with self.assertRaises(HistoryError):
            ScanHistory(cooldown=-1.0)

    def test_recent_is_newest_first(self):
        history = ScanHistory(cooldown=0.0)
        for name in "abc":
            history.offer(detection(name))
        self.assertEqual([scan.payload for scan in history.recent(2)], ["c", "b"])

    def test_recent_of_zero_is_empty(self):
        history = ScanHistory(cooldown=0.0)
        history.offer(detection())
        self.assertEqual(history.recent(0), [])

    def test_counts_by_symbology(self):
        history = ScanHistory(cooldown=0.0)
        history.offer(detection("a", "QR"))
        history.offer(detection("b", "QR"))
        history.offer(detection("4006381333931", "EAN_13"))
        self.assertEqual(history.counts_by_symbology(), {"EAN_13": 1, "QR": 2})

    def test_counts_by_kind(self):
        history = ScanHistory(cooldown=0.0)
        history.offer(detection("https://example.com"))
        history.offer(detection("plain words"))
        self.assertEqual(history.counts_by_kind(), {"text": 1, "url": 1})

    def test_describe_mentions_payload_and_symbology(self):
        history = ScanHistory(cooldown=0.0)
        scan = history.offer(detection("https://example.com"))
        text = scan.describe()
        self.assertIn("https://example.com", text)
        self.assertIn("QR", text)


class ScanLogTests(unittest.TestCase):
    def test_header_and_row_are_written(self):
        history = ScanHistory(cooldown=0.0)
        scan = history.offer(detection("https://example.com"))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "scans.csv"
            with ScanLog(path) as log:
                log.write(scan)
            rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["payload"], "https://example.com")
        self.assertEqual(rows[0]["symbology"], "QR")
        self.assertEqual(list(rows[0].keys()), list(CSV_FIELDS))

    def test_appending_does_not_repeat_the_header(self):
        history = ScanHistory(cooldown=0.0)
        scan = history.offer(detection())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "scans.csv"
            for _ in range(2):
                with ScanLog(path, append=True) as log:
                    log.write(scan)
            lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
        self.assertEqual(len(lines), 3)  # one header, two rows

    def test_overwrite_mode_replaces_the_file(self):
        history = ScanHistory(cooldown=0.0)
        scan = history.offer(detection())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "scans.csv"
            for _ in range(2):
                with ScanLog(path, append=False) as log:
                    log.write(scan)
            lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
        self.assertEqual(len(lines), 2)

    def test_nothing_is_created_when_nothing_is_scanned(self):
        """A run that scans nothing should not litter an empty file."""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "scans.csv"
            with ScanLog(path):
                pass
            self.assertFalse(path.exists())

    def test_missing_parent_directory_is_created(self):
        history = ScanHistory(cooldown=0.0)
        scan = history.offer(detection())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "deep" / "nested" / "scans.csv"
            with ScanLog(path) as log:
                log.write(scan)
            self.assertTrue(path.exists())

    def test_unwritable_path_raises_history_error(self):
        history = ScanHistory(cooldown=0.0)
        scan = history.offer(detection())
        with tempfile.TemporaryDirectory() as folder:
            blocker = Path(folder) / "blocker"
            blocker.write_text("not a directory", encoding="utf-8")
            with self.assertRaises(HistoryError):
                ScanLog(blocker / "scans.csv").write(scan)

    def test_rows_are_flushed_as_they_are_written(self):
        """A log that loses the last entries on a crash is not a log."""
        history = ScanHistory(cooldown=0.0)
        scan = history.offer(detection())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "scans.csv"
            log = ScanLog(path)
            log.write(scan)
            self.assertIn("QR", path.read_text(encoding="utf-8"))
            log.close()


# --------------------------------------------------------------------------
# Scene synthesis
# --------------------------------------------------------------------------


class SynthTests(unittest.TestCase):
    def test_place_centres_the_code(self):
        scene = qr_frame()
        centre = scene.placements[0].centre
        self.assertAlmostEqual(centre[0], FRAME[0] / 2.0, delta=2.0)
        self.assertAlmostEqual(centre[1], FRAME[1] / 2.0, delta=2.0)

    def test_place_rejects_a_code_larger_than_the_frame(self):
        with self.assertRaises(ConfigurationError):
            synth.place(render_qr("hello", module_size=40), width=100, height=100)

    def test_place_rejects_a_non_positive_scale(self):
        with self.assertRaises(ConfigurationError):
            synth.place(render_qr("hello"), scale=0.0)

    def test_grid_places_every_code(self):
        codes = [render_qr(f"code-{index}") for index in range(4)]
        scene = synth.grid(codes, width=1600, height=900, columns=2)
        self.assertEqual(len(scene.placements), 4)
        self.assertEqual(scene.payloads, {f"code-{index}" for index in range(4)})

    def test_grid_needs_at_least_one_code(self):
        with self.assertRaises(ConfigurationError):
            synth.grid([])

    def test_grid_placements_do_not_overlap(self):
        codes = [render_qr(f"code-{index}") for index in range(4)]
        scene = synth.grid(codes, width=1600, height=900, columns=2)
        boxes = []
        for item in scene.placements:
            xs, ys = item.corners[:, 0], item.corners[:, 1]
            boxes.append((xs.min(), ys.min(), xs.max(), ys.max()))
        for index, first in enumerate(boxes):
            for second in boxes[index + 1 :]:
                overlap_x = min(first[2], second[2]) - max(first[0], second[0])
                overlap_y = min(first[3], second[3]) - max(first[1], second[1])
                self.assertTrue(overlap_x <= 0 or overlap_y <= 0)

    def test_rotation_moves_the_ground_truth_with_the_image(self):
        scene = synth.rotate(qr_frame(), 90.0)
        self.assertFalse(np.allclose(scene.placements[0].corners, qr_frame().placements[0].corners))

    def test_rotation_by_a_full_turn_changes_nothing(self):
        scene = qr_frame()
        self.assertIs(synth.rotate(scene, 360.0), scene)

    def test_shrink_scales_corners_and_margin(self):
        original = qr_frame()
        scene = synth.shrink(original, 0.5)
        self.assertEqual(scene.image.shape[1], FRAME[0] // 2)
        self.assertAlmostEqual(
            scene.placements[0].margin[0], original.placements[0].margin[0] / 2.0, delta=0.5
        )

    def test_shrink_above_one_is_a_no_op(self):
        scene = qr_frame()
        self.assertIs(synth.shrink(scene, 1.0), scene)

    def test_perspective_rejects_an_extreme_strength(self):
        with self.assertRaises(ConfigurationError):
            synth.perspective(qr_frame(), 0.9)

    def test_blur_of_zero_is_a_no_op(self):
        scene = qr_frame()
        self.assertIs(synth.blur(scene, 0), scene)

    def test_noise_changes_pixels_but_not_truth(self):
        scene = qr_frame()
        noisy = synth.noise(scene, 20.0)
        self.assertFalse(np.array_equal(scene.image, noisy.image))
        self.assertEqual(noisy.payloads, scene.payloads)

    def test_noise_is_reproducible_for_a_seed(self):
        first = synth.noise(qr_frame(), 20.0, seed=7)
        second = synth.noise(qr_frame(), 20.0, seed=7)
        self.assertTrue(np.array_equal(first.image, second.image))

    def test_occlusion_region_is_validated(self):
        with self.assertRaises(ConfigurationError):
            synth.occlude(qr_frame(), 0.1, corner="middle")

    def test_occlusion_fraction_is_validated(self):
        with self.assertRaises(ConfigurationError):
            synth.occlude(qr_frame(), 1.0)

    def test_centre_occlusion_leaves_the_corners_alone(self):
        """Finder patterns must survive, or the test measures the wrong thing."""
        scene = synth.occlude(qr_frame(), 0.2, corner="centre")
        left, top, right, bottom = scene.placements[0].symbol_box
        patch = scene.image[int(top) : int(top) + 10, int(left) : int(left) + 10]
        self.assertIn(0, np.unique(patch))  # the finder pattern is still dark

    def test_corner_occlusion_erases_the_finder_pattern(self):
        scene = synth.occlude(qr_frame(), 0.2, corner="tl")
        left, top, _, _ = scene.placements[0].symbol_box
        patch = scene.image[int(top) + 2 : int(top) + 10, int(left) + 2 : int(left) + 10]
        self.assertNotIn(0, np.unique(patch))

    def test_centre_occlusion_defeats_decoding_before_detection(self):
        """Detection survives damage that decoding does not -- the whole point."""
        scene = synth.occlude(qr_frame(), 0.15, corner="centre")
        found = build_reader(barcode=False).read(scene.image)
        self.assertTrue(found)
        self.assertFalse(any(item.decoded for item in found))


# --------------------------------------------------------------------------
# Overlay
# --------------------------------------------------------------------------


class OverlayTests(unittest.TestCase):
    def test_annotate_returns_a_colour_copy_without_touching_the_input(self):
        frame = np.full((240, 320, 3), 60, dtype=np.uint8)
        canvas = annotate(frame, [], HudState(fps=30.0))
        self.assertEqual(canvas.shape, frame.shape)
        self.assertTrue(np.all(frame == 60))

    def test_greyscale_input_becomes_colour(self):
        canvas = annotate(np.zeros((240, 320), dtype=np.uint8), [], HudState())
        self.assertEqual(canvas.ndim, 3)

    def test_detection_outline_is_drawn(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        canvas = annotate(frame, [detection("https://example.com")], HudState())
        self.assertGreater(int(canvas.sum()), 0)

    def test_label_off_the_top_edge_does_not_raise(self):
        frame = np.zeros((60, 320, 3), dtype=np.uint8)
        draw_label(frame, "near the edge", (0, 2))

    def test_label_past_the_right_edge_does_not_raise(self):
        frame = np.zeros((60, 80, 3), dtype=np.uint8)
        draw_label(frame, "a very long label indeed", (70, 40))

    def test_empty_label_is_skipped(self):
        frame = np.zeros((60, 80, 3), dtype=np.uint8)
        draw_label(frame, "", (10, 40))
        self.assertEqual(int(frame.sum()), 0)

    def test_hud_shows_the_recent_list(self):
        history = ScanHistory(cooldown=0.0)
        history.offer(detection("https://example.com"))
        canvas = annotate(
            np.zeros((480, 640, 3), dtype=np.uint8), [], HudState(), history.recent()
        )
        self.assertGreater(int(canvas.sum()), 0)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


class ArgumentTests(unittest.TestCase):
    def test_parse_size(self):
        self.assertEqual(parse_size("1280x720"), (1280, 720))

    def test_parse_size_rejects_nonsense(self):
        for bad in ["1280", "axb", "0x720", "-1x10"]:
            with self.assertRaises(Exception):
                parse_size(bad)

    def test_scale_factor_bounds(self):
        self.assertEqual(scale_factor("2"), 2.0)
        for bad in ["0", "20", "nan"]:
            with self.assertRaises(Exception):
                scale_factor(bad)

    def test_zoom_fraction_bounds(self):
        self.assertEqual(zoom_fraction("0.3"), 0.3)
        for bad in ["0", "1.5", "-0.2"]:
            with self.assertRaises(Exception):
                zoom_fraction(bad)

    def test_defaults(self):
        args = build_parser().parse_args([])
        self.assertEqual(args.camera, 0)
        self.assertEqual(args.detector, "aruco")
        self.assertTrue(args.scan_qr)
        self.assertTrue(args.scan_barcode)

    def test_scanning_for_nothing_is_rejected(self):
        args = build_parser().parse_args(["--no-qr", "--no-barcode"])
        with self.assertRaises(ConfigurationError):
            resolve_config(args)

    def test_image_and_video_together_is_rejected(self):
        args = build_parser().parse_args(["--image", "a.png", "--video", "b.mp4"])
        with self.assertRaises(ConfigurationError):
            resolve_config(args)

    def test_missing_input_is_reported(self):
        args = build_parser().parse_args(["--image", "does-not-exist.png"])
        with self.assertRaises(ConfigurationError) as caught:
            resolve_config(args)
        self.assertIn("not found", str(caught.exception))

    def test_non_image_passed_to_image_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "clip.mp4"
            path.write_bytes(b"not really a video")
            args = build_parser().parse_args(["--image", str(path)])
            with self.assertRaises(ConfigurationError):
                resolve_config(args)

    def test_no_csv_clears_the_log_path(self):
        args = build_parser().parse_args(["--no-csv"])
        self.assertIsNone(resolve_config(args).csv_path)


class FpsMeterTests(unittest.TestCase):
    def test_no_reading_before_two_frames(self):
        self.assertEqual(FpsMeter().value, 0.0)

    def test_rate_from_a_fake_clock(self):
        clock = FakeClock()
        meter = FpsMeter(window=10, clock=clock)
        for _ in range(5):
            meter.tick()
            clock.advance(0.05)
        self.assertAlmostEqual(meter.value, 20.0, delta=0.5)

    def test_window_is_bounded(self):
        clock = FakeClock()
        meter = FpsMeter(window=3, clock=clock)
        for _ in range(50):
            meter.tick()
            clock.advance(0.01)
        self.assertAlmostEqual(meter.value, 100.0, delta=5.0)

    def test_a_stopped_clock_does_not_divide_by_zero(self):
        meter = FpsMeter(clock=FakeClock())
        meter.tick()
        meter.tick()
        self.assertEqual(meter.value, 0.0)


class ImageRunTests(unittest.TestCase):
    """End-to-end over the real CLI, on a rendered file."""

    def _write(self, folder: str, scene: synth.Scene, name: str = "frame.png") -> Path:
        path = Path(folder) / name
        self.assertTrue(cv2.imwrite(str(path), scene.image))
        return path

    def test_scanning_an_image_reports_the_payload(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self._write(folder, qr_frame("https://example.com/scan"))
            csv_path = Path(folder) / "scans.csv"
            with quiet() as output:
                code = main(
                    ["--image", str(path), "--no-display", "--csv", str(csv_path)]
                )
            self.assertEqual(code, 0)
            self.assertIn("https://example.com/scan", output.getvalue())
            self.assertIn("1 distinct code", output.getvalue())
            self.assertTrue(csv_path.exists())
            rows = list(csv.DictReader(csv_path.read_text(encoding="utf-8").splitlines()))
            self.assertEqual(rows[0]["payload"], "https://example.com/scan")

    def test_scanning_a_barcode_image(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self._write(folder, ean_frame("400638133393"))
            with quiet() as output:
                self.assertEqual(
                    main(["--image", str(path), "--no-display", "--no-csv"]), 0
                )
            self.assertIn("4006381333931", output.getvalue())

    def test_a_frame_with_several_codes(self):
        codes = [
            render_qr("https://example.com/scan"),
            render_qr("WIFI:T:WPA;S:Workshop;P:secret;;"),
            render_barcode("400638133393", module_width=4, height=276),
        ]
        with tempfile.TemporaryDirectory() as folder:
            path = self._write(
                folder, synth.grid(codes, width=1800, height=1200, columns=2)
            )
            with quiet() as output:
                self.assertEqual(
                    main(["--image", str(path), "--no-display", "--no-csv"]), 0
                )
            text = output.getvalue()
            self.assertIn("3 distinct code", text)
            self.assertIn("wifi", text)

    def test_a_blank_image_says_nothing_was_found(self):
        with tempfile.TemporaryDirectory() as folder:
            blank = synth.Scene(np.full((480, 640, 3), 235, dtype=np.uint8))
            path = self._write(folder, blank)
            with quiet() as output:
                self.assertEqual(
                    main(["--image", str(path), "--no-display", "--no-csv"]), 0
                )
            self.assertIn("No codes found", output.getvalue())

    def test_an_undecodable_code_is_reported_as_detected(self):
        scene = synth.occlude(qr_frame(), 0.15, corner="centre")
        with tempfile.TemporaryDirectory() as folder:
            path = self._write(folder, scene)
            with quiet() as output:
                self.assertEqual(
                    main(["--image", str(path), "--no-display", "--no-csv"]), 0
                )
            self.assertIn("decoded none", output.getvalue())

    def test_a_corrupt_file_is_reported_not_crashed(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "broken.png"
            path.write_bytes(b"this is not a PNG")
            with quiet() as output:
                self.assertEqual(
                    main(["--image", str(path), "--no-display", "--no-csv"]), 2
                )
            self.assertIn("could not read", output.getvalue())

    def test_annotated_frame_is_written_when_asked(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self._write(folder, qr_frame())
            out_dir = Path(folder) / "annotated"
            with quiet():
                self.assertEqual(
                    main(
                        [
                            "--image",
                            str(path),
                            "--no-display",
                            "--no-csv",
                            "--save-frames",
                            str(out_dir),
                        ]
                    ),
                    0,
                )
            self.assertTrue(list(out_dir.glob("*.png")))

    def test_missing_file_exits_with_an_error(self):
        with quiet() as output:
            self.assertEqual(main(["--image", "nope.png", "--no-display"]), 2)
        self.assertIn("not found", output.getvalue())


class VideoRunTests(unittest.TestCase):
    def test_scanning_a_video_of_rendered_frames(self):
        """The video loop, the cooldown and the CSV together."""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "clip.avi"
            scene = qr_frame("https://example.com/scan")
            height, width = scene.image.shape[:2]
            writer = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (width, height)
            )
            self.assertTrue(writer.isOpened())
            for _ in range(12):
                writer.write(scene.image)
            writer.release()

            csv_path = Path(folder) / "scans.csv"
            with quiet() as output:
                code = main(
                    [
                        "--video",
                        str(path),
                        "--no-display",
                        "--csv",
                        str(csv_path),
                        "--cooldown",
                        "600",
                    ]
                )
            self.assertEqual(code, 0)
            text = output.getvalue()
            # Twelve frames of the same code must log exactly one scan.
            self.assertIn("1 distinct code", text)
            self.assertIn("Suppressed", text)
            rows = list(csv.DictReader(csv_path.read_text(encoding="utf-8").splitlines()))
            self.assertEqual(len(rows), 1)

    def test_a_missing_video_is_reported(self):
        with quiet() as output:
            self.assertEqual(main(["--video", "nope.mp4", "--no-display"]), 2)
        self.assertIn("not found", output.getvalue())


if __name__ == "__main__":
    unittest.main()
