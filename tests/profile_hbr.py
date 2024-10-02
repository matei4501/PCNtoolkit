import os

os.environ["PYTENSOR_FLAGS"] = "floatX=float32"

from pcntoolkit.normative_model.norm_utils import norm_init
from warnings import filterwarnings

filterwarnings("ignore")
import pymc as pm
import pytensor
import pickle


# pytensor.config.profile = True
pytensor.config.profile_memory = False
pytensor.config.profile_optimizer = False
# pytensor.config.profiling__output = (
#     "/home/guus/Projects/PCNtoolkit/tests/profiling_output.txt"
# )
# pytensor.config.profiling__destination = "stdout"


def main():

    ########################### Experiment Settings ###############################

    random_state = 29

    working_dir = (
        "/home/guus/tmp"  # Specify a working directory to save data and results.
    )

    simulation_method = "linear"
    n_features = 1  # The number of input features of X
    n_grps = 10  # Number of batches in data
    n_samples = 2500  # Number of samples in each group (use a list for different
    # sample numbers across different batches)

    model_type = "bspline"  #  modelto try 'linear, ''polynomial', 'bspline'

    savedir = "/home/guus/Desktop/pcn_profile_data"

    ############################## Data Simulation ################################

    def load_data():
        with open(os.path.join(savedir, "train_data"), "rb") as f:
            X_train, Y_train, grp_id_train = pickle.load(f)
        with open(os.path.join(savedir, "test_data"), "rb") as f:
            X_test, Y_test, grp_id_test = pickle.load(f)
        return X_train, Y_train, grp_id_train, X_test, Y_test, grp_id_test

    X_train, Y_train, grp_id_train, X_test, Y_test, grp_id_test = load_data()

    model_confs = {
        # "M1": {
        #     "random_intercept_mu": "False",
        #     "linear_sigma": "False",
        #     "likelihood": "Normal",
        # },
        # "M2": {
        #     "random_intercept_mu": "True",
        #     "linear_sigma": "False",
        #     "likelihood": "Normal",
        # },
        # "M3": {
        #     "random_intercept_mu": "True",
        #     "linear_sigma": "False",
        #     "likelihood": "SHASHo",
        # },
        "M4": {
            "random_intercept_mu": "True",
            "linear_sigma": "False",
            "likelihood": "SHASHb",
        },
        "M5": {
            "random_intercept_mu": "True",
            "linear_sigma": "False",
            "likelihood": "SHASHb",
        },
    }

    for model_name, model_conf in model_confs.items():
        pytensor.config.profiling__destination = f"/home/guus/Desktop/pcn_profile_data/v30/{model_name}/profiling_nutpie_output.txt"
        print(f"Model: {model_name}")
        print("Initializing the model")
        nm = norm_init(
            X_train,
            Y_train,
            alg="hbr",
            model_type=model_type,
            linear_mu="True",
            random_slope_mu="False",
            random_sigma="False",
            linear_epsilon="False",
            linear_delta="False",
            **model_conf,
        )

        os.makedirs(os.path.join(savedir, "v30", model_name), exist_ok=True)

        print("Getting the model")
        model = nm.hbr.get_model(X_train, Y_train, grp_id_train)
        with model:
            idata = pm.sample(
                1000,
                tune=500,
                chains=4,
                cores=4,
                compute_convergence_checks=False,
                progressbar=True,
                nuts_sampler="nutpie",
                init="jitter+adapt_diag_grad",
            )


if __name__ == "__main__":
    main()
