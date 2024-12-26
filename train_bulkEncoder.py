import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from _misc import export_mtx


def train_BulkEncoder(
    epoch,
    model,
    GMVAE_model,
    max_epochs,
    optimizer,
    dataloader,
    scMus,
    scLogVars,
    scPis,
    device="cuda"
):

    model.train()
    model = model.to(device)
    GMVAE_model.eval()
    GMVAE_model = GMVAE_model.to(device)

    _sc_data = np.empty((0, 32738))
    _pbulk_data = np.empty((0, 32738))

    for _, (data) in enumerate(dataloader):
        bulk_data = data[0].to(device)

        mus, logvars, pis = model(bulk_data)

        mus = mus.squeeze()
        logvars = logvars.squeeze()
        pis = pis.squeeze()

        mus_loss = F.mse_loss(mus, scMus)
        logvars_loss = F.mse_loss(logvars, scLogVars)
        pis_loss = F.mse_loss(pis, scPis)

        loss = mus_loss + logvars_loss + pis_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(
            "Epoch[{}/{}]: mus_loss:{:.3f}, vars_loss:{:.3f}, pis_loss:{:.3f}".format(
                epoch + 1,
                max_epochs,
                mus_loss.item(),
                logvars_loss.item(),
                # h0_loss.item(),
                pis_loss.item(),
            )
        )

    if (epoch + 1) % 500 == 0:
        torch.save(model.state_dict(), f"_pt/bulkEncoder_model.pt")
