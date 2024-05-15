import ml_collections

def get_b16_config():
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 1
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    #config.resnet.att_type = 'CBAM'
    config.classifier = 'seg'
    config.representation_size = None
    config.resnet_pretrained_path = None
    #config.pretrained_path = '../model/vit_checkpoint/imagenet21k/ViT-B_16.npz'
    config.patch_size = 16

    config.decoder_channels = (256, 128, 64, 16)
    config.n_classes = 10
    config.activation = 'softmax'
    return config
def get_config():
    """Returns the Resnet50 + ViT-B/16 configuration."""
    config = get_b16_config()
    config.data=ml_collections.ConfigDict()
    config.data.img_size=512
    config.data.in_chans=3
    config.patches.grid = (16, 16)
    config.resnet = ml_collections.ConfigDict()
    config.resnet.num_layers = (3, 4, 6,3)
    config.resnet.width_factor = 0.5

    config.classifier = 'seg'
    config.trans = ml_collections.ConfigDict()
    config.trans.num_heads = [3, 6, 12, 24]
    config.trans.depths = [2, 2, 6, 2]
    config.trans.embed_dim = 96
    config.trans.window_size = 8

    # config.pretrained_path = '../model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz' #yuxunlian
    config.decoder_channels = (
    512, 256, 128, 64)  # (256,128,64,16)##1024,512,256,128,64)#(2048,1024,512,256,128)#(256, 128, 64, 16)
    config.skip_channels = [512, 256, 128,
                            64]  # [256,128,64,16]#[512,256,128,64,16]#[512,256,128,64,32]#[1024,512,256,128,64]#[512, 256, 64, 16]
    config.n_classes = 10
    config.n_skip = 3
    config.activation = 'softmax'
    # config.pretrained_path=

    return config