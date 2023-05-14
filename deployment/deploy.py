import argparse
import logging
from servicefoundry import Build, PythonBuild, Service, Resources, Port

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--workspace_fqn", required=True, type=str)
args = parser.parse_args()

service = Service(
    name="fastapi",
    image=Build(
        build_spec=PythonBuild(
            command="uvicorn app:app --port 8000 --host 0.0.0.0",
            requirements_path="requirements.txt",
        )
    ),
    ports=[
        Port(
            port=8000,
            host="https://intern-abhijit.demo1.truefoundry.com/predict/"
        )
    ],
    resources=Resources(
        cpu_request=0.5,
        cpu_limit=1,
        memory_request=1000,
        memory_limit=1500
    ),
    env={
        "UVICORN_WEB_CONCURRENCY": "1",
        "ENVIRONMENT": "dev"
    }
)
service.deploy(workspace_fqn=args.workspace_fqn)
