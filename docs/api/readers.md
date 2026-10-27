# Readers

Read one acceleration component per file: PEER NGA `.AT2`, RENAC (IG-EPN,
Ecuador) `.txt`, or generic text. Most code builds a `Record` with
[`Record.from_files`](record.md) instead of calling these directly.

::: apeQuake.readers.readers
    options:
      members:
        - RecordFile
        - read_record_file
        - detect_format
        - read_at2
        - read_renac
        - read_txt

::: apeQuake.core.units
    options:
      members:
        - normalize_unit
        - conversion_factor
