# Data

The analysis uses **Facebook100**: the complete intra-school Facebook friendship graphs of
100 US colleges and universities in September 2005, released by Traud, Mucha and Porter.

The data is **not stored in this repository**. Get it with:

```bash
make data          # or: uv run fb100 data
```

This downloads `facebook100.zip` (207 MB) from the Internet Archive mirror
<https://archive.org/details/oxford-2005-facebook-matrix>, checks its MD5
(`a7687ec2362d369803a3463e2884dbe8`), extracts the 100 school files to
`data/raw/facebook100/` (gitignored) and deletes the archive. Running it again is a no-op.

## Contents

Each `<School><id>.mat` file holds:

| Variable | Meaning |
|---|---|
| `A` | sparse symmetric adjacency matrix (friendships inside the school) |
| `local_info` | one row per node, 7 columns in this order: status flag, gender, major, second major/minor, dorm/house, year, high school |

Missing values are coded `0`. Node identities are anonymised integers; attribute values
are anonymised codes (the year column holds calendar years).

## Terms of use

- The original authors ask that any use cite
  A. L. Traud, P. J. Mucha, M. A. Porter, *Social Structure of Facebook Networks*,
  Physica A 391(16), 4165–4180 (2012), [arXiv:1102.2166](https://arxiv.org/abs/1102.2166).
- The dataset describes real people. It is used here only for aggregate, school-level
  statistics; no attempt is made to identify individuals, and none should be.
- The authors no longer distribute the files from their own pages. This repository
  therefore does not redistribute them and only points to the public archive.
