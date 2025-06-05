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

from mcmc.experiment_main import run_experiment
from mcmc.logreg_utils import eval_gt_logreg, get_gt_logreg, get_model_and_data
from mcmc.metrics import adjust_max_len
from mcmc.progressive import (
    ProgressiveEvaluator,
    ProgressiveLMC,
    ProgressiveLogger,
    ProgressiveNUTS,
)
from mcmc.progressive.progressive_plotting import make_figs


warnings.simplefilter("ignore", FutureWarning)

jnp.set_printoptions(precision=3, suppress=True, threshold=sys.maxsize)
print(jax.devices("cuda"))

dataset = scipy.io.loadmat("mcmc_data/benchmarks.mat")
names = [
    # "tbp",
    # "isolet_ab",
    # "banana",
    # "breast_cancer",
    # "diabetis",
    "flare_solar",
    # "german",
    # "heart",
    # "image",
    # "ringnorm",
    # "splice",
    # "thyroid",
    # "titanic",
    # "twonorm",
    # "waveform",
]


timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
prev_result_quic = lambda name: f"progressive_results/flare_solar_*.pkl"
prev_result_quic_adap = lambda name: f"progressive_results/flare_solar_*.pkl"
prev_result_nuts = lambda name: f"progressive_results/flare_solar_*.pkl"
prev_results_ubu = lambda name: f"progressive_results/flare_solar_*.pkl"
# prev_result_quic = lambda name: f"progressive_results/good_results/{name}_*.pkl"
# prev_result_nuts = lambda name: f"progressive_results/good_results/{name}_*.pkl"

evaluator = ProgressiveEvaluator()
logger = ProgressiveLogger(log_filename=f"progressive_results/log_{timestamp}.txt")
logger.start_log(timestamp)

PRIOR_START = False

nuts_warmup = 20

get_result_filename = (
    lambda name: f"progressive_results/{name}_{timestamp}.pkl"
)


def make_pid(atol, dt0):
    return diffrax.PIDController(
        atol=atol,
        rtol=0.0,
        dtmax=dt0 * 10,
        dtmin=dt0 / 10,
        pcoeff=0.1,
        icoeff=0.4,
    )


quic_kwargs = {
    "chain_len": 2**6,
    "chain_sep": 1.0,
    "dt0": 0.07,
    "solver": diffrax.QUICSORT(0.1),
    "pid": None,
    "prior_start": PRIOR_START,
}


quic_adap_kwargs = {
    "chain_len": 2**6,
    "chain_sep": 1.0,
    "dt0": 0.07,
    "solver": diffrax.QUICSORT(0.1),
    "pid": make_pid(0.1, 0.07),
    "prior_start": PRIOR_START,
}


euler_kwargs = {
    "chain_len": 2**6,
    "chain_sep": 0.5,
    "dt0": 0.03,
    "solver": diffrax.Euler(),
    "pid": make_pid(0.1, 0.03),
    "prior_start": PRIOR_START,
}
euler = ProgressiveLMC(euler_kwargs)

ubu_kwargs = {
    "chain_len": 2**6,
    "chain_sep": 1.0,
    "dt0": 0.035,
    "solver": custom_solvers.UBU(0.1),
    "pid": None,
    "prior_start": PRIOR_START,
}

dt0s = {
    "banana": 0.04,
    "splice": 0.01,
    "flare_solar": 0.05,
    "isolet_ab": 0.001,
}
seps = {
    "banana": 0.3,
    "splice": 0.5,
    "flare_solar": 3.0,
    "image": 1.0,
    "waveform": 1.0,
    "isolet_ab": 0.2,
}
atols = {
    "flare_solar": 0.3,
    "isolet_ab": 50.0,
    "banana": 0.04,
    "splice": 0.4,
    "image": 1.0,
    "titanic": 0.1,
}

nuts_lens = {
    "isolet_ab": 1000 * 2**5,
    "flare_solar": 2**8,
}

for name in names:
    model, model_args, test_args = get_model_and_data(dataset, name)
    data_dim = model_args[0].shape[1] + 1
    num_particles = 2**15 if name=="isolet_ab" else 2**15
    num_particles = adjust_max_len(num_particles, data_dim)
    config = {
        "num_particles": num_particles,
        "test_args": test_args,
    }

    # NUTS settings
    nuts_len = nuts_lens.get(name, 2**6)
    nuts = ProgressiveNUTS(
        nuts_warmup,
        nuts_len,
        prior_start=PRIOR_START,
        # get_previous_result_filename=prev_result_nuts,
    )

    # LMC settings
    quic_dt0 = dt0s.get(name, 0.07)
    chain_sep = seps.get(name, 0.5)
    atol = atols.get(name, 1.0)
    quic_kwargs["dt0"], quic_kwargs["chain_sep"] = quic_dt0, chain_sep
    quic_kwargs["pid"] = None
    quic_adap_kwargs["dt0"], quic_adap_kwargs["chain_sep"] = quic_dt0, chain_sep
    quic_adap_kwargs["pid"] = make_pid(atol, quic_dt0)
    euler_kwargs["dt0"] = quic_dt0 / 20
    euler_kwargs["chain_sep"] = chain_sep
    euler_kwargs["pid"] = make_pid(atol, quic_dt0 / 20)
    ubu_kwargs["dt0"] = quic_dt0 / 2
    ubu_kwargs["chain_sep"] = chain_sep
    ubu_kwargs["pid"] = None

    quic = ProgressiveLMC(
        quic_kwargs,
        # get_previous_result_filename=prev_result_quic
    )
    quic_adap = ProgressiveLMC(
        quic_adap_kwargs,
        # get_previous_result_filename=prev_result_quic_adap,
    )
    ubu = ProgressiveLMC(
        ubu_kwargs,
        # get_previous_result_filename=prev_results_ubu,
    )

    methods = [
        nuts,
        ubu,
        quic,
        quic_adap,
    ]

    logger.start_model_section(name)
    quic_atol_str = f"atol={atol}, "
    logger.print_log(
        f"NUTS(warmup={nuts.num_warmup}, total={nuts.chain_len}),"
        f" QUICSORT_ADAP({quic_atol_str}dt0={quic_dt0}, sep={chain_sep}),"
        f" QUICSORT(dt0={quic_dt0}, sep={chain_sep}),"
        f" UBU(dt0={quic_dt0 / 2}, sep={chain_sep}),"
        f" prior_start = {PRIOR_START}\n"
    )

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
        eval_gt_logreg,
        get_result_filename,
    )

    result_filename = get_result_filename(name)
    figs = make_figs(
        result_filename, save_name=f"progressive_results/plots/{name}_{timestamp}.pdf"
    )
