import os
from fireworks import Firework, Workflow, LaunchPad, ScriptTask
from os import listdir

# credentials pulled from env instead of hardcoded, set these before running:
#   export LEGACYPM_MONGO_HOST=mongodb05.nersc.gov
#   export LEGACYPM_MONGO_USER=legacypm_admin
#   export LEGACYPM_MONGO_PASSWORD=...
launchpad = LaunchPad(
    host=os.environ['LEGACYPM_MONGO_HOST'],
    username=os.environ['LEGACYPM_MONGO_USER'],
    name='legacypm',
    password=os.environ['LEGACYPM_MONGO_PASSWORD'],
)
# launchpad.reset('2025-01-22')

def create_firework(script_file, parameters, fw_name, parents):
    script_directory = "../../py_files/"
    task = ScriptTask(
        script=f"python {script_directory + script_file} " + parameters
    )
    return Firework(task, name=fw_name, parents=parents)

# brick_file = "./brick_list.txt"
degree = "029"

base_dirs = {
    "new_dir": f"/pscratch/sd/n/nelfalou/legacypm-catalogue/{ degree }/",
    "old_dir": f"/pscratch/sd/d/dstn/forced-motions-dr10/forced-brick/{ degree }/",
    "tractor_dir": f"/pscratch/sd/d/dstn/forced-motions-dr10/forced-brick/{ degree }/"
}

forced_base = "forced-{}.fits"
tractor_base = "tractor-forced-{}.fits"

old_files = listdir(base_dirs["old_dir"])
finished_files = listdir(base_dirs["new_dir"])
bricks = []
for file in old_files:
    if file[0] == 'f' and file not in finished_files:
        bricks.append(file[-13:-5])

fireworks = []
for brick in bricks:
    forced_fname = forced_base.format(brick)
    dcr_fw = create_firework(
        script_file="brick_DCR.py",
        parameters=f"{forced_fname} {base_dirs['new_dir']} {base_dirs['old_dir']}",
        fw_name=f"DCR_correction_{brick}",
        parents=None
    )
    ringmaps_fw = create_firework(
        script_file="brick_ringmaps.py",
        parameters=f"{forced_fname} {base_dirs['new_dir']} {base_dirs['new_dir']}",
        fw_name=f"Ringmaps_correction_{brick}",
        parents=[dcr_fw]
    )
    lateralmaps_fw = create_firework(
        script_file="brick_lateralmaps.py",
        parameters=f"{forced_fname} {base_dirs['new_dir']} {base_dirs['new_dir']}  {base_dirs['tractor_dir']}",
        fw_name=f"Lateralmaps_correction_{brick}",
        parents=[ringmaps_fw]
    )
    catalog_fw = create_firework(
        script_file="create_brick_catalogue.py",
        parameters=f"{forced_fname} {base_dirs['new_dir']} {base_dirs['new_dir']} {base_dirs['tractor_dir']}",
        fw_name=f"Catalogue_creation_{brick}",
        parents=[lateralmaps_fw]
    )
    fireworks.extend([dcr_fw, ringmaps_fw, lateralmaps_fw, catalog_fw])

workflow = Workflow(fireworks, name=f"degree { degree }")
launchpad.add_wf(workflow)
print(f"Added {len(fireworks)} fireworks for degree {degree} to the launchpad.")
print("Submit them with: qlaunch -r rapidfire -m <max_jobs> --nlaunches infinite")
