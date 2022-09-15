import hydra
import hydra.utils as hu
import pytorch_lightning as pl


@hydra.main(config_path="configs/", config_name="tagging")
def main(cfg):
    if "seed" in cfg:
        pl.seed_everything(cfg.seed)

    feature_extractor = hu.instantiate(cfg.features)
    fe_model = hu.instantiate(cfg.fe_model)
    net = hu.instantiate(
        cfg.net,
        front_end_model=fe_model
    )
    optimizer = hu.instantiate(cfg.optim, params=net.parameters())
    lr_scheduler = hu.instantiate(
        cfg.lr_scheduler, optimizer) if "lr_scheduler" in cfg else None
    criterion = hu.instantiate(cfg.criterion)

    datamodule = hu.instantiate(cfg.datamodule)
    model = hu.instantiate(
        cfg.model,
        net=net,
        feature_extractor=feature_extractor,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        criterion=criterion,
        datamodule=datamodule
    )

    model_ckpt = hu.instantiate(cfg.model_checkpoint)
    logger, callbacks = [], []
    profiler = None
    if "profiler" in cfg.trainer and cfg.trainer.profiler:
        profiler = pl.profiler.AdvancedProfiler(dirpath=cfg.logger.save_dir,
                                                filename=cfg.experiment)
    if "logger" in cfg:
        logger = hu.instantiate(cfg.logger)
    if "callbacks" in cfg:
        for _, cb_cfg in cfg.callbacks.items():
            callbacks.append(hu.instantiate(cb_cfg))

    if "resume" in cfg:
        trainer = hu.instantiate(cfg.trainer,
                                 checkpoint_callback=model_ckpt,
                                 callbacks=callbacks,
                                 logger=logger,
                                 resume_from_checkpoint=cfg.resume.ckpt_path,
                                 profiler=profiler)
        print("Resuming model checkpoint..")
    else:
        trainer = hu.instantiate(cfg.trainer,
                                 checkpoint_callback=model_ckpt,
                                 callbacks=callbacks,
                                 logger=logger,
                                 profiler=profiler)

    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
