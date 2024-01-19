## Launching Virtual Accelerator

### 1. Create config files

To see help:
```bash
   python input_maker.py -h
```
Create default config files for MEBT -> HEBT1
```bash
   python input_maker.py
```
Create config files for MEBT
```bash
   python input_maker.py --file MEBT_config.json MEBT
```

### 2. Run Virtual Accelerator

To see help:
```bash
   python virtual_accelerator.py -h
```

Run default MEBT -> HEBT1
```bash
   python virtual_accelerator.py
```

Run MEBT only (with printing all PVs)
```bash
   python virtual_accelerator.py --debug --bunch MEBT_in.dat MEBT
```