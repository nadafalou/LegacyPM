from fireworks import Firework, Workflow, LaunchPad, ScriptTask
from fireworks.core.rocket_launcher import rapidfire


launchpad = LaunchPad(host='mongodb05.nersc.gov', username='legacypm_admin', name='legacypm', password='legacypm')
# launchpad.reset('2024-11-28')

def create_firework(script_file, parameters, fw_name, parents):
    script_directory = "../../py/"
    task = ScriptTask(
        script=f"python {script_directory + script_file} " + parameters
    )
    return Firework(task, name=fw_name, parents=parents)

brick_file = "./brick_list.txt"
base_dirs = {
    "new_dir": "/pscratch/sd/n/nelfalou/fw-test/", 
    "old_dir": "/pscratch/sd/d/dstn/forced-nomovegaia/",
    "tractor_dir": "/pscratch/sd/d/dstn/forced-nomovegaia/"
    
}
forced_base = "forced-brickwise-{}.fits"
tractor_base = "tractor-forced-{}.fits"

# with open(brick_file, 'r') as f:
#     bricks = [line.strip() for line in f]
bricks = ["0292p045"]

# create Fireworks for each step
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
    
#     ringmaps_fw.spec["_dependencies"] = [dcr_fw.fw_id]
#     lateralmaps_fw.spec["_dependencies"] = [ringmaps_fw.fw_id]
#     catalog_fw.spec["_dependencies"] = [lateralmaps_fw.fw_id]
    
    fireworks.extend([dcr_fw, ringmaps_fw, lateralmaps_fw, catalog_fw])

workflow = Workflow(fireworks)
launchpad.add_wf(workflow)
rapidfire(launchpad)





# from fireworks import Firework, Workflow, LaunchPad
# from fireworks.core.rocket_launcher import rapidfire
# from fireworks.utilities.fw_utilities import explicit_serialize
# from fireworks.core.firework import FWAction
# import os
# import importlib.util
# import logging


# @explicit_serialize
# class SystematicCorrectionTask(FireTaskBase):
#     """
#     FireTask to apply a specific systematic correction to a given brick.
#     """
#     required_params = ["brick", "systematic", "function_file"]
#     optional_params = ["output_dir"]

#     def run_task(self, fw_spec):
#         brick = self["brick"]
#         systematic = self["systematic"]
#         function_file = self["function_file"]
#         output_dir = self.get("output_dir", "./corrected_data")

#         # Create output directory if it doesn't exist
#         os.makedirs(output_dir, exist_ok=True)
#         output_file = os.path.join(output_dir, f"{brick}_{systematic}.txt")

#         # Dynamically import the systematic correction function
#         spec = importlib.util.spec_from_file_location(systematic, function_file)
#         module = importlib.util.module_from_spec(spec)
#         spec.loader.exec_module(module)

#         # Assume the function to call has the same name as the systematic
#         correction_function = getattr(module, systematic)

#         # Run the correction function
#         print(f"Applying {systematic} to {brick}")
#         correction_function(brick, output_file)

#         return FWAction()
    
    
# @explicit_serialize
# class CatalogGenerationTask(FireTaskBase):
#     """
#     FireTask to generate the final catalog for a brick after all systematic corrections are complete.
#     """
#     required_params = ["brick", "catalog_script"]
#     optional_params = ["output_dir", "log_file"]

#     def run_task(self, fw_spec):
#         log_file = self.get("log_file", "progress.log")
#         logging.basicConfig(
#             filename=log_file,
#             level=logging.INFO,
#             format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
#         )
#         logger = logging.getLogger(f"{self['brick']}_catalog")

#         brick = self["brick"]
#         catalog_script = os.path.join(scripts_directory, self["catalog_script"])
#         output_dir = self.get("output_dir", "./catalogs")

#         try:
#             os.makedirs(output_dir, exist_ok=True)
#             output_file = os.path.join(output_dir, f"{brick}_catalog.txt")
#             logger.info(f"Starting catalog generation for {brick}")

#             # Command to execute the catalog generation script
#             cmd = f"python {catalog_script} --brick {brick} --output {output_file}"
#             logger.info(f"Running command: {cmd}")
#             os.system(cmd)

#             logger.info(f"Completed catalog generation for {brick}. Output: {output_file}")

#         except Exception as e:
#             logger.error(f"Failed to generate catalog for {brick}: {str(e)}")
#             raise

#         return FWAction()
    

# if __name__ == "__main__":
    
#     launchpad = LaunchPad(host='mongodb05.nersc.gov', username='legacypm_admin', name='legacypm', password='legacypm')
#     # launchpad.reset('2024-11-28')

#     # brick_list_file = "brick_list.txt"
#     scripts_directory = "../py/"
#     systematic_files = {
#         "DCR": "brick_DCR.py",
#         "RM": "brick_ringmaps.py",
#         "LM": "brick_lateralmaps.py"
#     }

#     # with open(brick_list_file, "r") as f:
#     #     bricks = [line.strip() for line in f.readlines()]
#     bricks = ["0292p045"]

#     fireworks = []
#     for brick in bricks:
#         previous_fw = None
#         for systematic, function_file in systematic_files.items():
#             fw = Firework(
#                 SystematicCorrectionTask(
#                     brick=brick,
#                     systematic=systematic,
#                     function_file=function_file,
#                     log_file=f"logs/{brick}_{systematic}.log"
#                 ),
#                 name=f"{brick}_{systematic}",
#                 parents=previous_fw,
#             )
#             fireworks.append(fw)
#             previous_fw = fw

#         catalog_fw = Firework(
#             CatalogGenerationTask(
#                 brick=brick,
#                 catalog_script=catalog_script,
#                 log_file=f"logs/{brick}_catalog.log"
#             ),
#             name=f"{brick}_catalog",
#             parents=previous_fw,  # This ensures it runs after all systematic corrections
#         )
#         fireworks.append(catalog_fw)

#     wf = Workflow(fireworks, name="SystematicCorrectionsAndCatalogWorkflow")
#     launchpad.add_wf(wf)

#     rapidfire(launchpad)