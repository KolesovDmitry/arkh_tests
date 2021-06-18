import os

from tensorflow.python.tools import saved_model_utils
  
# Import, authenticate and initialize the Earth Engine library.
import ee
ee.Authenticate()
ee.Initialize()


PROJECT = 'gee4avral'
REGION = 'us-central1'

MODEL_DIR = 'models'
MODEL_VERSION = 'v0_0_2'
MODEL_NAME = 'model_' + MODEL_VERSION
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

# Put the EEified model next to the trained model directory.
EEIFIED_DIR = 'gs://nextgis_gee_avral/SentinelData/Models/EEfied_' + MODEL_NAME


meta_graph_def = saved_model_utils.get_meta_graph_def(MODEL_PATH, 'serve')
inputs = meta_graph_def.signature_def['serving_default'].inputs
outputs = meta_graph_def.signature_def['serving_default'].outputs

# Just get the first thing(s) from the serving signature def.  i.e. this
# model only has a single input and a single output.
input_name = None
for k,v in inputs.items():
  input_name = v.name
  break

output_name = None
for k,v in outputs.items():
  output_name = v.name
  break

# Make a dictionary that maps Earth Engine outputs and inputs to
# AI Platform inputs and outputs, respectively.
import json
input_dict = "'" + json.dumps({input_name: "array"}) + "'"
output_dict = "'" + json.dumps({output_name: "output"}) + "'"
print(input_dict)
print(output_dict)


# You need to set the project before using the model prepare command.
!earthengine set_project {PROJECT}
!earthengine model prepare --source_dir {MODEL_PATH} --dest_dir {EEIFIED_DIR} --input {input_dict} --output  {output_dict}


MODEL_NAME = 'L8_S2_change_probas'
VERSION_NAME = MODEL_VERSION

!gcloud ai-platform models create {MODEL_NAME} \
 --project {PROJECT} \
 --region {REGION}

!gcloud ai-platform versions create {VERSION_NAME} \
  --project {PROJECT} \
  --region {REGION} \
  --model {MODEL_NAME} \
  --origin {EEIFIED_DIR} \
  --framework "TENSORFLOW" \
  --runtime-version=2.3 \
  --python-version=3.7
