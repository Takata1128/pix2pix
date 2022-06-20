from config import Config
from trainer import Trainer

if __name__ == "__main__":
    config = Config()
    trainer = Trainer(config)
    trainer.train()
