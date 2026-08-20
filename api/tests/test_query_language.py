from unittest import TestCase
from unittest.mock import patch

from engine.query_language import resolve_queries


class ResolveQueriesTests(TestCase):
    def test_uses_supplied_english_without_translation(self):
        with patch("engine.query_language.translate_vi_to_en") as translate:
            resolved = resolve_queries("  xuất khẩu gạo  ", "  rice export  ")

        translate.assert_not_called()
        self.assertEqual(resolved.query_vi, "xuất khẩu gạo")
        self.assertEqual(resolved.query_en, "rice export")
        self.assertEqual(resolved.query_en_source, "user")

    def test_translates_when_english_is_blank(self):
        with patch(
            "engine.query_language.translate_vi_to_en",
            return_value="Vietnam rice export",
        ) as translate:
            resolved = resolve_queries("xuất khẩu gạo", "  ")

        translate.assert_called_once_with("xuất khẩu gạo")
        self.assertEqual(resolved.query_en, "Vietnam rice export")
        self.assertEqual(resolved.query_en_source, "translated")
