import datetime as dt
import uuid

from ai_platform.run_ai_job import run_ai_job
from ai_platform.serializers import CreativeScoreRequest
from parser import parse_args
import omegaconf
import yaml
import sys


def main(args):
    with open("conf/base/parameters.yml", "r") as f:
        cfg = omegaconf.DictConfig(yaml.safe_load(f))

    run_uuid = args.run_uuid or str(uuid.uuid4())

    cs_request = CreativeScoreRequest(
        run_uuid=run_uuid,
        pipeline=args.pipeline,
        list_asset_type=args.list_asset_type,
        list_industry_name=args.list_industry_name,
        ad_started_at=dt.datetime.strptime(args.ad_started_at, "%Y-%m-%d"),
        ad_ended_at=dt.datetime.strptime(args.ad_ended_at, "%Y-%m-%d"),
        y_colname=args.y_colname,
        objective_family=args.objective_family,
        model_perf_metric=args.model_perf_metric,
        skip_data_download=args.skip_data_download,
        population_size=args.population_size,
        population_decay=args.population_decay,
        max_generation=args.max_generation,
        varcomb_max_comb=args.varcomb_max_comb,
        varcomb_max_colinearity=args.varcomb_max_colinearity,
    )
    success, returncode = run_ai_job(
        job_config=cfg.cloud.ai_platform_job_config, job_request=cs_request, job_purpose="creascore"
    )
    if not success:
        print(returncode)


args = parse_args(args=sys.argv[1:])
if __name__ == "__main__":
    main(args)
