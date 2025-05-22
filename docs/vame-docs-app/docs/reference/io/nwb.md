---
sidebar_label: nwb
title: io.nwb
---

#### logger\_config

#### logger

#### get\_base\_nwbfile

```python
def get_base_nwbfile(session_name: str, ds: xr.Dataset, nwbfile_kwargs: dict,
                     subject_kwargs: dict)
```

Create base NWBFile object.

#### export\_to\_nwb

```python
def export_to_nwb(config: dict,
                  nwbfile_kwargs: Optional[list[dict]] = None,
                  subject_kwargs: Optional[list[dict]] = None)
```

Export VAME project to NWB format.

