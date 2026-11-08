# Rename to bandedqr

The active project is now `bandedqr`. This is a clean break before publication,
not an alias release. The numerical algorithms, mathematical class names
(`BandQR`, `BatchedBandQR`, and the other factorization classes), and version
`0.2.0` are unchanged.

## Code and configuration

- Install the distribution `bandedqr` and use `import bandedqr` or
  `from bandedqr import solve`.
- The standalone artifact is `bandedqr_standalone.py`.
- Native headers, shared-library filenames, and exported function prefixes now
  use `bandedqr`. Rebuild native consumers against the new headers and library.
- Configuration variables and build macros use `BANDEDQR_`, including
  `BANDEDQR_LIBRARY`, `BANDEDQR_CACHE`, `BANDEDQR_LOCAL_DATA_ROOT`,
  `BANDEDQR_THREAD_MIN_KL`, `BANDEDQR_THREAD_MIN_WORK`, `BANDEDQR_THREAD_MAX`,
  and `BANDEDQR_NO_THREADS`.
- Old imports, native symbol names, and `BANDQR_` variables are not aliases.
  No command-line executable was installed by the package before this rename.

For an existing editable installation, uninstall the old distribution and
install this checkout under the new name using the same Python interpreter:

```sh
python -m pip uninstall bandqr
python -m pip install --no-deps --no-build-isolation -e .
```

The project's standard Mac interpreter is
`~/miniforge3/envs/dedalus3/bin/python`; its environment name is unchanged.

## Working copies and private synchronization

- Mac checkout: `~/repos/bandedqr`
- Rusty checkout: `~/projects/bandedqr`
- Rusty bare repository: `~/git/bandedqr.git`
- Mac remote `rusty`: `rusty:/mnt/home/sholden/git/bandedqr.git`
- Rusty remote `origin`: `/mnt/home/sholden/git/bandedqr.git`

No GitHub repository or PyPI release is created by this migration. Public
project URLs should be added only after a hosting repository is selected.

## Existing environments and historical evidence

Do not move a virtual environment just to rename it: launchers can contain
absolute paths. Existing Rusty environments remain at their original paths.
Future environment bootstraps default to `~/envs/bandedqr/cpu`. To use the
existing CPU interpreter without creating another environment, explicitly pass
`--environment /mnt/home/sholden/envs/bandqr/cpu` to the Rusty submitter. A
submission allocates a Slurm job; the rename itself does not submit one.

Future Rusty campaign data defaults to `~/ceph/bandedqr/`, and future Mac
conformance data defaults to `~/bandedqr-data/`. Existing `~/ceph/bandqr/`,
`~/bandqr-data/`, and campaign records under `runs/` are historical evidence:
leave their bytes, commit IDs, recorded paths, and source snapshots unchanged.
The containing checkout moves, but its retained run records are not rewritten.

The archived Dedalus integration, earlier changelog entries, and retained
figure PDFs describe earlier `bandqr` work. They intentionally retain the old
name. The current figure-generation source uses `bandedqr`; do not rerun timing
experiments merely to relabel historical plots.
