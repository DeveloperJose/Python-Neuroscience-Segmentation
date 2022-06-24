"""
Authors: Bibek Aryal, Alex Arnal, Jose Perez
"""
from segmentation.data.data import fetch_loaders
from segmentation.model.frame import Framework
import segmentation.model.functions as fn
import keys

import yaml, json, pathlib, warnings, pdb, torch, logging, time
from torch.utils.tensorboard import SummaryWriter
from addict import Dict
from twilio.rest import Client
import numpy as np

warnings.filterwarnings("ignore")
if __name__ == "__main__":
    client = Client(keys.account_sid, keys.auth_token)

    conf = Dict(yaml.safe_load(open('./conf/unet_train.yaml')))
    data_dir = pathlib.Path(conf.data_dir)
    class_name = conf.class_name
    run_name = conf.run_name
    processed_dir = data_dir / "processed"
    loaders = fetch_loaders(processed_dir, conf.batch_size, conf.use_channels)
    loss_fn = fn.get_loss(conf.model_opts.args.outchannels, conf.loss_opts)            
    log_dir = pathlib.Path(conf.log_dir) / run_name
    early_stopping = conf.early_stopping
    frame = Framework(
        loss_fn = loss_fn,
        model_opts=conf.model_opts,
        optimizer_opts=conf.optim_opts,
        reg_opts=conf.reg_opts,
        device=conf.device
    )

    assert not conf.fine_tune, 'Fine-Tuning is currently not supported'
    # if conf.fine_tune:
    #     fn.log(logging.INFO, f"Finetuning the model")
    #     run_name = conf.run_name+"_finetuned"
    #     model_path = f"{data_dir}/runs/{conf.run_name}/models/model_bestVal.pt"
    #     if torch.cuda.is_available():
    #         state_dict = torch.load(model_path)
    #     else:
    #         state_dict = torch.load(model_path, map_location="cpu")
    #     frame.load_state_dict(state_dict)
    #     #frame.freeze_layers()

    # Setup logging
    writer = SummaryWriter(log_dir / 'logs')
    writer.add_text("Configuration Parameters", json.dumps(conf))
    #writer.add_graph(frame.get_model(), next(iter(loaders['train'])))
    out_dir = log_dir / 'models'
    val_loss = np.inf
    
    fn.print_conf(conf)
    fn.log(logging.INFO, "# Training Instances = {}, # Validation Instances = {}".format(len(loaders["train"]), len(loaders["val"])))
    allValLosses = []
    # best_val_loss = np.inf
    best_val_iou = 0
    best_epoch = 0
    best_metrics = None
    for epoch in range(conf.epochs):
        # train loop
        loss = {}
        start = time.time()
        loss["train"], train_metric = fn.train_epoch(epoch, loaders["train"], frame, conf)
        fn.log_metrics(writer, train_metric, epoch+1, "train", conf.log_opts.mask_names)
        train_time = time.time() - start
        
        # validation loop
        start = time.time()
        loss["val"], val_metric = fn.validate(epoch, loaders["val"], frame, conf)
        fn.log_metrics(writer, val_metric, epoch+1, "val", conf.log_opts.mask_names)
        val_iou = val_metric['IoU'][1].item()
        val_time = time.time() - start

        # if epoch % 5 == 0:
        #     fn.log(logging.INFO, 'Logging images')
        #     fn.log_images(writer, frame, loaders["train"], epoch, "train")
        #     fn.log_images(writer, frame, loaders["val"], epoch, "val")

        writer.add_scalars("Loss", loss, epoch)
        if epoch > 0 and loss['val'] < min(allValLosses):
            frame.save(out_dir, "bestVal")
            print("\n\nSaving bestVal Model at epoch %s\n\n"%epoch)
        allValLosses.append(loss['val'])
        
        fn.print_metrics(conf, train_metric, val_metric)
        writer.flush()

        # Early Stopping
        diff = val_iou - best_val_iou
        if diff > 0.001:
            best_val_iou = val_iou
            best_epoch = epoch
            best_metrics = [train_metric, val_metric]
        else:
            epochs_without_improving = epoch - best_epoch
            if epochs_without_improving < early_stopping:
                fn.log(logging.INFO, f'\tVal iou {val_iou:.4f} did not improve from {best_val_iou:.4f} | {diff:.4f} | {epochs_without_improving} epochs without improvement | Patience remaining: {early_stopping-epochs_without_improving}')
            else:
                fn.log(logging.INFO, f'\tVal iou {val_iou:.4f} did not improve from {best_val_iou:.4f}, patience ran out so stopping early')
                break

    fn.log(logging.INFO, f'Saving final. Last val_time was {val_time}s')
    frame.save(out_dir, "final")
    writer.close()

    #%% Send a text message via Twilio
    client.messages.create(
        body=f'{run_name} has completed with best val_iou {best_val_iou} in epoch {best_epoch} after {epoch+1} epochs | {best_metrics}',
        from_=keys.src_phone,
        to=keys.dst_phone
    )