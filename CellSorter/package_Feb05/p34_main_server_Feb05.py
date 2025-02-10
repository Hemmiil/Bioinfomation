from package_Feb05 import p02_Exp
import json
import time

def main():
    p02_instance = p02_Exp.Experiment()
    p02_instance.get_data()
    
    config_path = "Exp_configs/config_Feb09.json"

    with open(config_path, "r") as f:
        configs = json.load(f)

    n_samples = configs["n_samples"]
    for i in range(n_samples):
        start = time.time()
        
        config = {
            "PCA_random_state": configs["PCA_random_state"][i],
            "GMM_random_state": configs["GMM_random_state"][i],
            "GMM_num_clusters": configs["GMM_num_clusters"][i],
        }

        X_cluster, X_raw_cluster = p02_instance.experiment(
            config=config,
        )

        p02_instance.output_X(X_cluster=X_cluster, X_raw_cluster=X_raw_cluster, config=config)
        p02_instance.output_statistics(X_cluster=X_cluster, config=config)

        end = time.time()

        done_time = end - start

        print(done_time)

if __name__ == "__main__":
    main()