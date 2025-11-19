from qiskit_ibm_runtime import QiskitRuntimeService

token = "d23b9Oeubuhm0iLkosV15yECIpweC93em7eGUR7IFsHf"
service = QiskitRuntimeService.save_account(
    token=token, # IBM Cloud API key.
    # Your token is confidential. Do not share your token in public code.
    instance="crn:v1:bluemix:public:quantum-computing:us-east:a/259760873b384161b757348f60bffda7:432c6064-683b-4cc4-87fd-c76df37db799::", # Optionally specify the instance to use.
    # Additionally, instances of a certain plan type are excluded if the plan name is not specified.
    region="us-east", # Optionally set the region to prioritize. Accepted values are 'us-east' or 'eu-de'. This is ignored if the instance is specified.
    set_as_default=True, # Optionally set these as your default credentials.
    )
 
# If you named your credentials, optionally specify the name here, as follows:
# QiskitRuntimeService(name='account-name')
# If you don't specify a name, the default credentials are loaded.
service = QiskitRuntimeService()