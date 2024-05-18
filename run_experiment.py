import random
from datetime import datetime

import hydra
import pandas as pd
from omegaconf import DictConfig
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score

from cayley_adam.stiefel_optimizer import AdamG
from definitions import *
from model import HeNCler
from utils import load_dataset, f1_score
from tqdm import tqdm

@hydra.main(version_base=None, config_path=str(ROOT_DIR / "conf"), config_name="default_config")
def run_experiment(cfg: DictConfig) -> None:
    seed = 1941488137
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    for run in range(1, cfg.num_runs+1):

        data, num_classes, num_features = load_dataset(cfg.d_name, cfg.standardize, cfg.to_undirected,
                                                       AddRandomWalkPE=True)

        if type(cfg.num_cl) is not int and cfg.num_cl.lower() == 'infer':
            cfg.num_cl = num_classes

        if type(cfg.s) is not int and cfg.s.lower() == 'infer':
            cfg.s = num_classes * 2

        model = HeNCler(input_dim=data.num_features,
                        hidden_dim=cfg.hidden_dim,
                        output_dim=cfg.output_dim,
                        num_cl=cfg.num_cl,
                        s=cfg.s
                        )

        params_stiefel, params_other = model.param_state()
        optimizer1 = AdamG([{'params': params_stiefel, 'lr': cfg.lrg, 'stiefel': True}])
        optimizer2 = torch.optim.Adam(params_other, lr=cfg.lr)

        model.train()
        best_nmi = 0
        best_f1 = 0

        start_time = datetime.now()
        for epoch in tqdm(range(1, cfg.epochs + 1), unit='epoch'):
            model.train()
            er, losses = model(data)
            pp_loss, node_rec_loss, edge_rec_loss, orto_loss = losses['pp_loss'], losses['node_rec_loss'], losses[
                'edge_rec_loss'], losses['orto_loss']
            loss = pp_loss.pow(2) + cfg.gamma_node_rec * node_rec_loss + cfg.gamma_edge_rec * edge_rec_loss

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            loss.backward()
            optimizer2.step()
            optimizer1.step()

            model.eval()
            with torch.no_grad():
                cluster_ids = KMeans(n_clusters=cfg.num_cl).fit_predict(er.detach().numpy())
                nmi = normalized_mutual_info_score(data.y.squeeze(), cluster_ids)
                f1 = f1_score(data.y, cluster_ids)
                if nmi > best_nmi: best_nmi = nmi
                if f1 > best_f1: best_f1 = f1

        best_dict = {'best_nmi': best_nmi * 100, 'best_f1': best_f1 * 100}
        formatters = {'nmi': lambda x: "{:.2f}%".format(x), 'f1': lambda x: "{:.2f}%".format(x)}

        print(pd.DataFrame({k: formatters[k.split('_')[1]](v) for k, v in best_dict.items()}, index=[f'run {run}']))

        best_dict.update({'time': (datetime.now() - start_time).seconds})

        if run == 1:
            df_results = pd.DataFrame(best_dict, index=[f'run {run}'])
        else:
            df_results = pd.concat([df_results, pd.DataFrame(best_dict, index=[f'run {run}'])])

    df_results.loc['mean', :] = df_results.mean()
    df_results.loc['std', :] = df_results.std()
    df_results = np.round(df_results, 2)
    print('\nResults')
    print(df_results)
    #df_results.to_csv(OUT_DIR / ('results_' + cfg.d_name))

if __name__ == "__main__":
    results = run_experiment()
