# Repository-local Kaggle CLI account

This project uses `scripts/kaggle.sh`, which sets `KAGGLE_CONFIG_DIR` to the
gitignored `.kaggle/` directory in this repository. It does not read or modify
the default global Kaggle CLI configuration.

1. In the Kaggle account that should own the private artifact dataset and
   kernel, create a new API token in **Settings → API → Create New Token**.
2. Save the downloaded file as `.kaggle/kaggle.json` in the repository root.
3. Restrict its permissions:

   ```bash
   mkdir -p .kaggle
   chmod 700 .kaggle
   chmod 600 .kaggle/kaggle.json
   ```

4. Use the wrapper rather than the global `kaggle` command:

   ```bash
   ./scripts/kaggle.sh datasets status kushchaudhari/ship-motion-reimplementation-artifacts
   ./scripts/kaggle.sh kernels push -p . --accelerator gpu --timeout 43200
   ```

The first account used for an existing private dataset or private kernel must
have access to those resources. To transfer ownership, create a new private
dataset/kernel from the new account and update `kaggle.yml` and
`kernel-metadata.json` with the new account slug.

The root metadata enables Internet because the notebook clones the public
repository during setup. To capture the live execution log without exposing
credentials, run:

```bash
python scripts/watch_kaggle_run.py --slug kushchaudhari/ship-motion-leakage-free-benchmark
```
