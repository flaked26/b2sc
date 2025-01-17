import torch
import torch.nn as nn
from utils import configure
import os
from torch.utils.data import TensorDataset, DataLoader
from models import GaussianMixtureVAE, bulkEncoder
from generate import generate

#from _test_recon import *
from _misc import *
from _validation import *

_retrain = True
_ori = True


# Train GMVAE. Refer to train_GMVAE.py for the implementation and model checkpoint path.
def train_model_GMVAE(
    x_train,
    labels_train,
    max_epochs,
    dataloader,
    proportion_tensor,
    mapping_dict,
    color_map,
    model_param_tuple,
    device="cuda",
):

    # Check if pre-trained weights are available.
    if (
        not _retrain
        and os.path.exists("_pt/GMVAE_mus.pt")
        and os.path.exists("_pt/GMVAE_logvars.pt")
        and os.path.exists("_pt/GMVAE_pis.pt")
    ):
        print("Pre-trained GMVAE_mus and GMVAE_logvars EXIST. Skipping training.")
        return 0
    else:
        print(
            f"Pre-trained GMVAE_mus and GMVAE_logvars DO NOT EXIST. Training for {max_epochs} epochs."
        )

        from models import GaussianMixtureVAE
        from train_GMVAE import train_GMVAE

        input_dim, hidden_dim, latent_dim, K = model_param_tuple
        GMVAE_model = GaussianMixtureVAE(input_dim, hidden_dim, latent_dim, K)
        optimizer = torch.optim.Adam(GMVAE_model.parameters(), lr=1e-3)
        print(f"Using {torch.cuda.device_count()} GPUs!")
        # Wrap the model with nn.DataParallel
        GMVAE_model = nn.DataParallel(GMVAE_model)
        try:
            # Load the state dict (assuming it was saved from a model wrapped with nn.DataParallel)
            if _ori:
                gmvae_state_dict = torch.load("saved_files/GMVAE_model.pt")
            else:
                gmvae_state_dict = torch.load("_pt/GMVAE_model.pt")
            GMVAE_model.load_state_dict(gmvae_state_dict, strict=True)
            print("Loaded existing GMVAE_model.pt")
        except:
            # Initialize weights.
            for m in GMVAE_model.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    nn.init.zeros_(m.bias)
            print("Initialized GMVAE_model")

        kl_weight = 0.0
        kl_weight_max = 1.0
        losses = []

        for epoch in range(0, max_epochs):
            kl_weight_increment = kl_weight_max / (100000)

            if kl_weight < kl_weight_max:
                kl_weight += kl_weight_increment
                kl_weight = min(kl_weight, kl_weight_max)
            # Train model.
            total_loss, mus, logvars, pis = train_GMVAE(
                x_train,
                labels_train,
                GMVAE_model,
                epoch,
                dataloader,
                optimizer,
                proportion_tensor,
                kl_weight,
                mapping_dict,
                color_map,
                max_epochs,
                device,
            )
            losses.append(total_loss)

        plt.figure(figsize=(10, 5))
        plt.plot(losses, label="Train Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Train Loss")
        plt.savefig(f"_plots/loss_GMVAE.png")
        plt.close()


# Train BulkEncoder. Refer to train_bulkEncoder.py for the implementation and model checkpoint path.
def train_model_BulkEncoder(
    max_epochs, dataloader, model_param_tuple, device="cuda", train_more=False
):
    # Check if pre-trained weights are available.
    if not _retrain and os.path.exists(f"_pt/bulkEncoder_model.pt"):
        if train_more:
            print(
                f"Pre-trained bulkEncoder_model EXIST. Additionally training for {max_epochs} epochs."
            )
        else:
            print("Pre-trained bulkEncoder_model EXIST. Skipping training.")
            return 0
    else:
        print(
            f"Pre-trained bulkEncoder_model DOES NOT exist. Training for {max_epochs} epochs."
        )

    from models import GaussianMixtureVAE, bulkEncoder
    from train_bulkEncoder import train_BulkEncoder

    scMus = (
        torch.load("saved_files/GMVAE_mus.pt").to(device).detach().requires_grad_(False)
        if _ori
        else torch.load("_pt/GMVAE_mus_mean.pt").to(device).detach().requires_grad_(False)
    )
    scLogVars = (
        torch.load("saved_files/GMVAE_logvars.pt").to(device).detach().requires_grad_(False)
        if _ori
        else torch.load("_pt/GMVAE_logvars_mean.pt").to(device).detach().requires_grad_(False)
    )
    scPis = (
        torch.load("saved_files/GMVAE_pis.pt").to(device).detach().requires_grad_(False)
        if _ori
        else torch.load("_pt/GMVAE_pis_mean.pt").to(device).detach().requires_grad_(False)
    )

    input_dim, hidden_dim, latent_dim, K = model_param_tuple
    bulkEncoder_model = bulkEncoder(input_dim, hidden_dim, latent_dim, K)

    if os.path.exists(f"_pt/bulkEncoder_model.pt"):
        if _ori:
            encoder_state_dict = torch.load(f"saved_files/bulkEncoder_model.pt")
        else:
            encoder_state_dict = torch.load(f"_pt/bulkEncoder_model.pt")
        bulkEncoder_model.load_state_dict(encoder_state_dict, strict=True)
    else:
        # Initialize weights.
        for m in bulkEncoder_model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    optimizer = torch.optim.Adam(bulkEncoder_model.parameters(), lr=1e-3)
    GMVAE_model = GaussianMixtureVAE(input_dim, hidden_dim, latent_dim, K)
    GMVAE_model = nn.DataParallel(GMVAE_model)

    # Load the state dict (assuming it was saved from a model wrapped with nn.DataParallel)
    if _ori:
        gmvae_state_dict = torch.load("saved_files/GMVAE_model.pt")
    else:
        gmvae_state_dict = torch.load("_pt/GMVAE_model.pt")
    GMVAE_model.load_state_dict(gmvae_state_dict, strict=True)
    bulkEncoder_model = bulkEncoder_model.to(device)

    for epoch in range(0, max_epochs):
        train_BulkEncoder(
            epoch,
            bulkEncoder_model,
            GMVAE_model,
            max_epochs,
            optimizer,
            dataloader,
            scMus,
            scLogVars,
            scPis,
            device,
        )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ############################ 0. Prepare args
    data_dir = "data/"
    barcode_path = data_dir + "barcode_to_celltype.csv"
    args = configure(data_dir, barcode_path, _retrain)

    ############################ 1. Train GMVAE for scMu and scLogVar.
    input_dim = args.input_dim
    hidden_dim = args.hidden_dim
    latent_dim = args.latent_dim
    K = args.K

    train_model_GMVAE(
        x_train=args.X_train,
        labels_train=args.labels_train,
        max_epochs=args.train_GMVAE_epochs,
        dataloader=args.dataloader,
        proportion_tensor=args.cell_type_fractions,
        mapping_dict=args.mapping_dict,
        color_map=args.color_map,
        model_param_tuple=(input_dim, hidden_dim, latent_dim, K),
        device=device,
    )

    ############################ 1.99 additional process for bulk data
    bulk_list = []
    for _, (data, label) in enumerate(args.dataloader):
        data = data.to(device)
        bulk_data = data.sum(dim=0)
        bulk_data = bulk_data.unsqueeze(0)
        bulk_list.append(bulk_data)
    bulk_tensor = torch.cat(bulk_list, dim=0)
    export_mtx(bulk_tensor.cpu().detach().numpy(), "bulk_train")
    bulk_dataset = TensorDataset(bulk_tensor)
    bulk_dataloader = DataLoader(bulk_dataset, batch_size=1, shuffle=True)

    ############################ 2. Train scDecoder for reconstruction using trained scMu and scLogVar.
    train_model_BulkEncoder(
        max_epochs=args.bulk_encoder_epochs,
        dataloader=bulk_dataloader,
        model_param_tuple=(input_dim, hidden_dim, latent_dim, K),
        device=device,
        train_more=False,
    )

    ############################ 3. Generate. Refer to generate.py for the implementation and data save path.

    # num_cells = args.num_cells
    num_cells = args.X_test.shape[0]

    # Load the state dict (assuming it was saved from a model wrapped with nn.DataParallel)
    GMVAE_model = GMVAE_model = GaussianMixtureVAE(input_dim, hidden_dim, latent_dim, K)
    GMVAE_model = nn.DataParallel(GMVAE_model)
    if _ori:
        gmvae_state_dict = torch.load("saved_files/GMVAE_model.pt")
    else:
        gmvae_state_dict = torch.load("_pt/GMVAE_model.pt")
    GMVAE_model.load_state_dict(gmvae_state_dict, strict=True)

    bulkEncoder_model = bulkEncoder(input_dim, hidden_dim, latent_dim, K)
    if _ori:
        encoder_state_dict = torch.load(f"saved_files/bulkEncoder_model.pt")
    else:
        encoder_state_dict = torch.load(f"_pt/bulkEncoder_model.pt")
    bulkEncoder_model.load_state_dict(encoder_state_dict, strict=True)

    _gen_opt = "test"
    if _gen_opt == "train":
        gen_dataloader = bulk_dataloader
    elif _gen_opt == "test":
        bulk_list = []
        for _, (data, label) in enumerate(args.testloader):
            data = data.to(device)
            bulk_data = data.sum(dim=0)
            bulk_data = bulk_data.unsqueeze(0)
            bulk_list.append(bulk_data)
        bulk_tensor = torch.cat(bulk_list, dim=0)
        export_mtx(bulk_tensor.cpu().detach().numpy(), "bulk_test")
        bulk_dataset = TensorDataset(bulk_tensor)
        gen_dataloader = DataLoader(bulk_dataset, batch_size=1, shuffle=True)

    generate(
        bulkEncoder_model,
        GMVAE_model,
        gen_dataloader,
        num_cells,
        device=device,
        opt=_gen_opt,
    )

    ############################ 4. Validation
    validate()
