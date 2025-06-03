import datetime
import sys
import warnings

import custom_solvers
import diffrax
import jax
import jax.numpy as jnp
import jax.random as jr
import scipy
from numpyro.infer import MCMC, NUTS, Predictive  # noqa: F401
import argparse

from mcmc.experiment_main import run_experiment
from mcmc.logreg_utils import eval_gt_logreg, get_gt_logreg, get_model_and_data
from mcmc.metrics import adjust_max_len
from mcmc.progressive import (
    ProgressiveEvaluator,
    ProgressiveLMC,
    ProgressiveLogger,
    ProgressiveNUTS,
)

parser = argparse.ArgumentParser()
parser.add_argument("--data_name", type=str, default="flare_solar", help="Name of the dataset to use")
args = parser.parse_args()

warnings.simplefilter("ignore", FutureWarning)

jnp.set_printoptions(precision=3, suppress=True, threshold=sys.maxsize)
print(jax.devices("cuda"))

dataset = scipy.io.loadmat("mcmc_data/benchmarks.mat")

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

get_result_filename = (
    lambda name: f"progressive_results/{name}_{timestamp}.pkl"
)

name = args.data_name
evaluator = ProgressiveEvaluator()
logger = ProgressiveLogger(log_filename=f"progressive_results/find_atols_log_{name}_{timestamp}.txt")
logger.start_log(timestamp)
logger.start_model_section(name)
model, model_args, test_args = get_model_and_data(dataset, name)
data_dim = model_args[0].shape[1] + 1
num_particles = adjust_max_len(2**15, data_dim)
config = {
    "num_particles": num_particles,
    "test_args": test_args,
}

# prev_result = lambda name: f"progressive_results/good_results/{name}_*.pkl"

pcoeff = 0.1
icoeff = 0.4

def make_pid(atol, dt0):
    return diffrax.PIDController(
        atol=atol,
        rtol=0.0,
        dtmax=dt0 * 10,
        dtmin=dt0 / 10,
        pcoeff=pcoeff,
        icoeff=icoeff,
    )


dt0s = {
    "banana": 0.04,
    "splice": 0.01,
    "flare_solar": 0.1,
    "isolet_ab": 0.001,
}
seps = {
    "banana": 0.3,
    "splice": 0.5,
    "flare_solar": 3.0,
    "image": 1.0,
    "waveform": 1.0,
    "isolet_ab": 0.5,
}

dt0 = dt0s.get(name, 0.07)
chain_sep = seps.get(name, 0.5)
logger.print_log(
    f"Using dt0={dt0}, chain_sep={chain_sep}, pcoeff={pcoeff}, icoeff={icoeff}, num_particles={num_particles} for {name}"
)

quic_adaptive_kwargs = {
    "chain_len": 2**5,
    "chain_sep": chain_sep,
    "dt0": dt0,
    "solver": diffrax.QUICSORT(0.1),
    "pid": make_pid(0.1, 0.07),
    "prior_start": False,
}

atols = [0.005 * 2**i for i in range(8)]

for atol in atols:
    quic_adaptive_kwargs["pid"] = make_pid(atol, dt0)
    logger.print_log(f"\nATOL={atol:.3f}")
    quic_adap = ProgressiveLMC(quic_adaptive_kwargs)
    methods = [quic_adap]

    run_experiment(
        jr.key(1),
        model,
        model_args,
        name,
        methods,
        config,
        evaluator,
        logger,
        get_gt_logreg,
        None,
        get_result_filename,
    )