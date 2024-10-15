import json
import unittest
from pathlib import Path
from unittest.mock import mock_open, patch

import yaml

from encourage.utils.file_manager import FileManager


class TestFileManager(unittest.TestCase):
    @patch("encourage.utils.file_manager.Path.mkdir")
    def setUp(self, mock_mkdir):
        self.filepath = Path("/fake/path/to/file.txt")
        self.file_manager = FileManager(self.filepath)
        mock_mkdir.assert_called_once()

    @patch("encourage.utils.file_manager.Path.unlink")
    @patch("encourage.utils.file_manager.Path.exists", return_value=True)
    def test_delete_file(self, mock_exists, mock_unlink):
        self.file_manager.delete()
        mock_exists.assert_called_once()
        mock_unlink.assert_called_once()

    @patch("encourage.utils.file_manager.Path.exists", return_value=False)
    def test_delete_file_not_exist(self, mock_exists):
        self.file_manager.delete()
        mock_exists.assert_called_once()

    @patch("builtins.open", new_callable=mock_open, read_data="file content")
    def test_read_file(self, mock_file):
        result = self.file_manager.read()
        mock_file.assert_called_once_with(self.filepath, "rt", encoding="utf-8")
        self.assertEqual(result, "file content")

    @patch("builtins.open", new_callable=mock_open)
    def test_write_file(self, mock_file):
        self.file_manager.write("new content")
        mock_file.assert_called_once_with(self.filepath, "wt", encoding="utf-8")
        mock_file().write.assert_called_once_with("new content")

    @patch("builtins.open", new_callable=mock_open)
    def test_append_file(self, mock_file):
        self.file_manager.append("appended content")
        mock_file.assert_called_once_with(self.filepath, "at", encoding="utf-8")
        mock_file().write.assert_called_once_with("appended content")

    @patch("builtins.open", new_callable=mock_open, read_data='{"key": "value"}')
    def test_load_json(self, mock_file):
        result = self.file_manager.load_json()
        mock_file.assert_called_once_with(self.filepath, "rt", encoding="utf-8")
        self.assertEqual(result, {"key": "value"})

    @patch("builtins.open", new_callable=mock_open)
    def test_dump_json(self, mock_file):
        data = {"key": "value"}
        self.file_manager.dump_json(data)
        mock_file.assert_called_once_with(self.filepath, "wt", encoding="utf-8")
        mock_file().write.assert_called_once_with(json.dumps(data, ensure_ascii=False))

    @patch("builtins.open", new_callable=mock_open, read_data="key: value\n")
    def test_load_yaml(self, mock_file):
        result = self.file_manager.load_yaml()
        mock_file.assert_called_once_with(self.filepath, "rt", encoding="utf-8")
        self.assertEqual(result, {"key": "value"})

    @patch("builtins.open", new_callable=mock_open)
    def test_dump_yaml(self, mock_file):
        data = {"key": "value"}
        self.file_manager.dump_yaml(data)
        mock_file.assert_called_once_with(self.filepath, "wt", encoding="utf-8")
        mock_file().write.assert_called_once_with(yaml.safe_dump(data))

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='{"key": "value"}\n{"key2": "value2"}',
    )
    def test_load_jsonlines(self, mock_file):
        result = self.file_manager.load_jsonlines()
        mock_file.assert_called_once_with(self.filepath, "rt", encoding="utf-8")
        self.assertEqual(result, [{"key": "value"}, {"key2": "value2"}])

    @patch("builtins.open", new_callable=mock_open)
    def test_dump_jsonlines(self, mock_file):
        data = [{"key": "value"}, {"key2": "value2"}]
        self.file_manager.dump_jsonlines(data)
        mock_file.assert_called_once_with(self.filepath, "wt", encoding="utf-8")
        expected = (
            json.dumps(data[0], ensure_ascii=False)
            + "\n"
            + json.dumps(data[1], ensure_ascii=False)
            + "\n"
        )
        mock_file().write.assert_called_once_with(expected)

    @patch("encourage.utils.file_manager.Path.exists", return_value=True)
    def test_file_exists(self, mock_exists):
        result = self.file_manager.file_exists()
        mock_exists.assert_called_once()
        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
