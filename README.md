#####
Attention: from the last block of unet
Similarity: random uniform

######  dataset and experimental setting
For emnist and omniglot
experiment: layer_size   --generator_layers = [64, 64, 128, 128]
                         --gen_inner_layers = [gen_depth_per_layer, gen_depth_per_layer,
                            gen_depth_per_layer, gen_depth_per_layer]
            image_size   --[28,28,1]

architecture 1conection: idx-2 > 0


script:CUDA_VISIBLE_DEVICES=1 nohup python -u train_dagan_with_matchingclassifier.py --dataset emnist --image_width 28 --batch_size 32 --generator_inner_layers 3 --discriminator_inner_layers 1 --num_generations 32 --experiment_title emnist1way3shotattention_2connection --num_of_gpus 1 --dropout_rate_value 0 --selected_classes 1 --support_number 3 --matching 1 --fce 0 --full_context_unroll_k 0 --is_z2_vae 0 --is_z2 0 --loss_G 0.1 --loss_D 1 --loss_KL 0 --loss_CLA 1 --loss_FSL 0 --loss_recons_B 0 --loss_matching_G 0 --loss_matching_D 10 --loss_sim 0 --z_dim 256 --strategy 1 > vaen2.log 2>&1 &


For vggface
experiment: layer_size   --generator_layers = [64, 64, 128, 128]
                         --gen_inner_layers = [gen_depth_per_layer, gen_depth_per_layer,
                            gen_depth_per_layer, gen_depth_per_layer]
            image_size   --[84,84,3]

architecture 1conection: idx-2 > 0
script:CUDA_VISIBLE_DEVICES=1 nohup python -u train_dagan_with_matchingclassifier.py --dataset vggface --image_width 128 --batch_size 25 --generator_inner_layers 3 --discriminator_inner_layers 1 --num_generations 32 --experiment_title vggface1way3shot_Noconnection --num_of_gpus 1 --dropout_rate_value 0 --selected_classes 1 --support_number 3 --matching 1 --fce 0 --full_context_unroll_k 0 --is_z2_vae 0 --is_z2 0 --loss_G 0.1 --loss_D 1 --loss_KL 0 --loss_CLA 1 --loss_FSL 0 --loss_recons_B 0 --loss_matching_G 0 --loss_matching_D 10 --loss_sim 0 --z_dim 256 --strategy 1 > vaen2.log 2>&1 &



For flowers& animals
experiment: layer_size --generator_layers = [64, 64, 128, 128, 256, 256]
                       --gen_inner_layers = [gen_depth_per_layer, gen_depth_per_layer,
                            gen_depth_per_layer, gen_depth_per_layer,gen_depth_per_layer, gen_depth_per_layer]
            image_size --[128,128,3]

architecture 1conection: idx-2 > 0
CUDA_VISIBLE_DEVICES=1 nohup python -u train_dagan_with_matchingclassifier.py --dataset flowers --image_width 128 --batch_size 20 --generator_inner_layers 3 --discriminator_inner_layers 1 --num_generations 32 --experiment_title RandomSim_flowers1way3shot6layersRandomTrainRandomTest_2connection --num_of_gpus 1 --dropout_rate_value 0 --selected_classes 1 --support_number 3 --matching 1 --fce 0 --full_context_unroll_k 0 --is_z2_vae 0 --is_z2 0 --loss_G 1 --loss_D 1 --loss_KL 0 --loss_CLA 1 --loss_FSL 0 --loss_recons_B 0 --loss_matching_G 0 --loss_matching_D 100 --loss_sim 0 --z_dim 256 --strategy 1 > vaen1.log 2>&1 &


###### To continue training the matchingGAN

####### 1 connection idx-2 >0  on emnist and omniglot

CUDA_VISIBLE_DEVICES=1 python train_dagan_with_matchingclassifier.py --dataset omniglot --image_width 28 --batch_size 32 --generator_inner_layers 3 --discriminator_inner_layers 1 --num_generations 32 --experiment_title Test_omniglot1way3shot --num_of_gpus 1 --dropout_rate_value 0 --selected_classes 1 --support_number 3 --matching 1 --fce 0 --full_context_unroll_k 0 --is_z2_vae 0 --is_z2 0 --loss_G 1 --loss_D 1 --loss_KL 0 --loss_CLA 1 --loss_FSL 0 --loss_recons_B 0.1 --loss_matching_G 0 --loss_matching_D 1 --loss_sim 0 --z_dim 256 --strategy 1 --restore_path ./RandomSim/omniglot1way3shot4layersRandomTrainRandomTest/2connection/saved_models/train_LOSS_z2vae0_z20_g1.0_d0.1_kl0.0_cla1.0_fzl_cla0.0_reconsB0.0_matchingG0.0_matchingD1.0_sim0.0_Net_batchsize32_zdim256_epoch115.ckpt --continue_from_epoch 115

CUDA_VISIBLE_DEVICES=1 python train_dagan_with_matchingclassifier.py --dataset emnist --image_width 28 --batch_size 32 --generator_inner_layers 3 --discriminator_inner_layers 1 --num_generations 32 --experiment_title Test_emnist1way1shot --num_of_gpus 1 --dropout_rate_value 0 --selected_classes 1 --support_number 1 --matching 1 --fce 0 --full_context_unroll_k 0 --is_z2_vae 0 --is_z2 0 --loss_G 1 --loss_D 1 --loss_KL 0 --loss_CLA 1 --loss_FSL 0 --loss_recons_B 0.1 --loss_matching_G 0 --loss_matching_D 1 --loss_sim 0 --z_dim 256 --strategy 1 --restore_path ./RandomSim/emnist1way3shot4layersRandomTrainRandomTest/2connection/saved_models/train_LOSS_z2vae0_z20_g1.0_d0.1_kl0.0_cla1.0_fzl_cla0.0_reconsB0.0_matchingG0.0_matchingD1.0_sim0.0_Net_batchsize32_zdim256_epoch10.ckpt --continue_from_epoch 11


#### 1 connection idx-2 > 0 on vggface
CUDA_VISIBLE_DEVICES=1 python train_dagan_with_matchingclassifier.py --dataset vggface --image_width 84 --batch_size 32 --generator_inner_layers 3 --discriminator_inner_layers 1 --num_generations 32 --experiment_title Test_vggface1way3shot --num_of_gpus 1 --dropout_rate_value 0 --selected_classes 1 --support_number 3 --matching 1 --fce 0 --full_context_unroll_k 0 --is_z2_vae 0 --is_z2 0 --loss_G 0.1 --loss_D 1 --loss_KL 0 --loss_CLA 1 --loss_FSL 0 --loss_recons_B 0 --loss_matching_G 0 --loss_matching_D 10 --loss_sim 0 --z_dim 256 --strategy 1 --restore_path ./RandomSim/vggface1way3shot4layersRandomTrainRandomTest/2connection/saved_models/train_LOSS_z2vae0_z20_g1.0_d0.1_kl0.0_cla1.0_fzl_cla0.0_reconsB1.0_matchingG0.0_matchingD1.0_sim0.0_Net_batchsize32_zdim256_epoch30.ckpt --continue_from_epoch 31



#### 2 connection idx-3 > 0 on flowers and animals
CUDA_VISIBLE_DEVICES=1 nohup python -u train_dagan_with_matchingclassifier.py --dataset flowers --image_width 128 --batch_size 20 --generator_inner_layers 3 --discriminator_inner_layers 1 --num_generations 32 --experiment_title RandomSim_flowers1way3shot6layersRandomTrainRandomTest_2connection --num_of_gpus 1 --dropout_rate_value 0 --selected_classes 1 --support_number 3 --matching 1 --fce 0 --full_context_unroll_k 0 --is_z2_vae 0 --is_z2 0 --loss_G 1 --loss_D 1 --loss_KL 0 --loss_CLA 1 --loss_FSL 0 --loss_recons_B 0 --loss_matching_G 0 --loss_matching_D 100 --loss_sim 0 --z_dim 256 --strategy 1 --restore_path ./RandomSim/flowers1way3shot6layersRandomTrainRandomTest/2connection/saved_models/train_LOSS_z2vae0_z20_g1.0_d1.0_kl0.0_cla1.0_fzl_cla0.0_reconsB0.0_matchingG0.0_matchingD1.0_sim0.0_Net_batchsize20_zdim256_epoch350.ckpt --continue_from_epoch 350 > vaen1.log 2>&1 &


###### To generate images from trained matchingGAN
####### data preparation
* selecting visualization images from the unseen categories, those selected images are stored into the './coarse-data/visualization_training_images/' and then using data_preparation.py to preprocess them into numpy array, eg [generate_image_label_pairs(dataroot=dataroot, dataset=dataset, image_size=84, channels=3, each_class_total_samples=3)]
* changing the array data root in the data_with_matchingclassifier.py
* runing  test_dagan_with_matchingclassifier_for_generation and soring the generated images in the path Test
* for generation images map and split it into single image by visluazation_images_selection.py



CUDA_VISIBLE_DEVICES=1 nohup python -u test_dagan_with_matchingclassifier_for_generation --dataset flowers --image_width 128 --batch_size 20 --generator_inner_layers 3 --discriminator_inner_layers 1 --num_generations 32 --experiment_title RandomSim_flowers1way3shot6layersRandomTrainRandomTest_2connection --num_of_gpus 1 --dropout_rate_value 0 --selected_classes 1 --support_number 3 --matching 1 --fce 0 --full_context_unroll_k 0 --is_z2_vae 0 --is_z2 0 --loss_G 1 --loss_D 1 --loss_KL 0 --loss_CLA 1 --loss_FSL 0 --loss_recons_B 0 --loss_matching_G 0 --loss_matching_D 100 --loss_sim 0 --z_dim 256 --strategy 1 --restore_path ./RandomSim/flowers1way3shot6layersRandomTrainRandomTest/2connection/saved_models/train_LOSS_z2vae0_z20_g1.0_d1.0_kl0.0_cla1.0_fzl_cla0.0_reconsB0.0_matchingG0.0_matchingD1.0_sim0.0_Net_batchsize20_zdim256_epoch350.ckpt --continue_from_epoch 350 > vaen1.log 2>&1 &



CUDA_VISIBLE_DEVICES=1 python test_dagan_with_matchingclassifier_for_generation.py --dataset omniglot --image_width 28 --batch_size 4 --generator_inner_layers 3 --discriminator_inner_layers 1 --num_generations 32 --experiment_title Test_omniglot1way1shot --num_of_gpus 1 --dropout_rate_value 0 --selected_classes 1 --support_number 1 --matching 1 --fce 0 --full_context_unroll_k 0 --is_z2_vae 0 --is_z2 0 --loss_G 1 --loss_D 1 --loss_KL 0 --loss_CLA 1 --loss_FSL 0 --loss_recons_B 0.1 --loss_matching_G 0 --loss_matching_D 1 --loss_sim 0 --z_dim 256 --strategy 1 --restore_path ./RandomSim/omniglot1way3shot4layersRandomTrainRandomTest/2connection/saved_models/train_LOSS_z2vae0_z20_g1.0_d0.1_kl0.0_cla1.0_fzl_cla0.0_reconsB0.0_matchingG0.0_matchingD1.0_sim0.0_Net_batchsize32_zdim256_epoch115.ckpt --continue_from_epoch 115



CUDA_VISIBLE_DEVICES=0 python test_dagan_with_matchingclassifier_for_generation.py --dataset emnist --image_width 28 --batch_size 4 --generator_inner_layers 3 --discriminator_inner_layers 1 --num_generations 32 --experiment_title Test_emnist1way3shot --num_of_gpus 1 --dropout_rate_value 0 --selected_classes 1 --support_number 3 --matching 1 --fce 0 --full_context_unroll_k 0 --is_z2_vae 0 --is_z2 0 --loss_G 1 --loss_D 1 --loss_KL 0 --loss_CLA 1 --loss_FSL 0 --loss_recons_B 0.1 --loss_matching_G 0 --loss_matching_D 1 --loss_sim 0 --z_dim 256 --strategy 1 --restore_path ./RandomSim/emnist1way3shot4layersRandomTrainRandomTest/2connection/saved_models/train_LOSS_z2vae0_z20_g1.0_d0.1_kl0.0_cla1.0_fzl_cla0.0_reconsB0.0_matchingG0.0_matchingD1.0_sim0.0_Net_batchsize32_zdim256_epoch10.ckpt --continue_from_epoch 11


CUDA_VISIBLE_DEVICES=0 python test_dagan_with_matchingclassifier_for_generation.py --dataset vggface --image_width 84 --batch_size 4 --generator_inner_layers 3 --discriminator_inner_layers 1 --num_generations 32 --experiment_title Test_vggface1way3shot --num_of_gpus 1 --dropout_rate_value 0 --selected_classes 1 --support_number 3 --matching 1 --fce 0 --full_context_unroll_k 0 --is_z2_vae 0 --is_z2 0 --loss_G 0.1 --loss_D 1 --loss_KL 0 --loss_CLA 1 --loss_FSL 0 --loss_recons_B 0 --loss_matching_G 0 --loss_matching_D 10 --loss_sim 0 --z_dim 256 --strategy 1 --restore_path ./RandomSim/vggface1way3shot4layersRandomTrainRandomTest/2connection/saved_models/train_LOSS_z2vae0_z20_g1.0_d0.1_kl0.0_cla1.0_fzl_cla0.0_reconsB1.0_matchingG0.0_matchingD1.0_sim0.0_Net_batchsize32_zdim256_epoch30.ckpt --continue_from_epoch 31







#### pretrain classifier
CUDA_VISIBLE_DEVICES=0  python pretrain_classifier.py --pretrain 1 --dataset omniglot --general_classification_samples 15 --is_training 1  --experiment_title AugmentedClassification_omniglot --image_width 28

CUDA_VISIBLE_DEVICES=0  python pretrain_classifier.py --pretrain 1 --dataset emnist --general_classification_samples 500 --is_training 1  --experiment_title AugmentedClassification_emnist --image_width 28

CUDA_VISIBLE_DEVICES=1  python pretrain_classifier.py --pretrain 1 --dataset vggface --general_classification_samples 80 --is_training 1  --experiment_title AugmentedClassification_vggface --image_width 64


#### training general classifier
checking data.py --> classes=testing.classes
restore_classifier_path --> pretrained on trianing classes and finetine on low data sampels(general classification baseline)

selected_classes:
omniglot: 212
emnist:
vggface:


CUDA_VISIBLE_DEVICES=1  python  train_dagan_augmented_general_fewshot_classification_with_matchingclassifier.py --dataset omniglot --image_width 28 --batch_size 32 --generator_inner_layers 3 --discriminator_inner_layers 3 --augmented_number 32 --is_fewshot_setting 0 --experiment_title AugmentedGeneralClassification_omniglot --num_of_gpus 1 --z_dim 256 --dropout_rate_value 0 --selected_classes 212 --support_number 3 --matching 1 --fce 0 --full_context_unroll_k 0 --average_per_class_embeddings 0 --general_classification_samples 5 --pretrained_epoch 0 --classification_total_epoch 100  --restore_path ./RandomSim/omniglot1way3shot4layersRandomTrainRandomTest/2connection/saved_models/train_LOSS_z2vae0_z20_g1.0_d0.1_kl0.0_cla1.0_fzl_cla0.0_reconsB0.0_matchingG0.0_matchingD1.0_sim0.0_Net_batchsize32_zdim256_epoch115.ckpt --restore_classifier_path 



CUDA_VISIBLE_DEVICES=0 python train_dagan_augmented_general_fewshot_classification_with_matchingclassifier.py --dataset vggface --image_width 84 --batch_size 32 --generator_inner_layers 3 --discriminator_inner_layers 3 --augmented_number 32 --is_fewshot_setting 0 --experiment_title AugmentedClassification_vggface --num_of_gpus 1 --z_dim 256 --dropout_rate_value 0 --selected_classes 20 --support_number 3 --matching 1 --fce 0 --full_context_unroll_k 0 --average_per_class_embeddings 0 --general_classification_samples 5 --pretrained_epoch 0 --classification_total_epoch 60 --continue_from_epoch 260 --restore_path ./RandomSim/vggface1way3shot4layersRandomTrainRandomTest/2connection/saved_models/train_LOSS_z2vae0_z20_g1.0_d0.1_kl0.0_cla1.0_fzl_cla0.0_reconsB1.0_matchingG0.0_matchingD1000.0_sim0.0_Net_batchsize32_zdim256_epoch25.ckpt --restore_classifier_path ./AugmentedClassification/vggface/saved_models/Pretrain_on_Source_Domain_50_0.1335565447807312.ckpt





#### training few-shot classifier
checking data.py --> classes=testing.classes
restore_classifier_path --> pretraining on training datasets


## 5 way 5 shot
CUDA_VISIBLE_DEVICES=0 python train_dagan_augmented_general_fewshot_classification_with_matchingclassifier.py --dataset omniglot --image_width 28 --batch_size 32 --generator_inner_layers 3 --discriminator_inner_layers 3 --augmented_number 0 --is_fewshot_setting 1 --experiment_title AugmentedClassification_omniglot --num_of_gpus 1 --z_dim 256 --dropout_rate_value 0 --selected_classes 5 --support_number 3 --matching 1 --fce 0 --full_context_unroll_k 0 --average_per_class_embeddings 0 --general_classification_samples 5 --pretrained_epoch 0 --classification_total_epoch 60 --restore_path ./RandomSim/omniglot1way3shot4layersRandomTrainRandomTest/2connection/saved_models/train_LOSS_z2vae0_z20_g1.0_d0.1_kl0.0_cla1.0_fzl_cla0.0_reconsB0.0_matchingG0.0_matchingD1.0_sim0.0_Net_batchsize32_zdim256_epoch115.ckpt --restore_classifier_path 



CUDA_VISIBLE_DEVICES=1 python train_dagan_augmented_general_fewshot_classification_with_matchingclassifier.py --dataset vggface --image_width 84 --batch_size 32 --generator_inner_layers 3 --discriminator_inner_layers 3 --augmented_number 512 --is_fewshot_setting 1 --experiment_title AugmentedClassification_vggface --num_of_gpus 1 --z_dim 256 --dropout_rate_value 0 --selected_classes 5 --support_number 3 --matching 1 --fce 0 --full_context_unroll_k 0 --average_per_class_embeddings 0 --general_classification_samples 5 --pretrained_epoch 0 --classification_total_epoch 60 --continue_from_epoch 260 --restore_path ./RandomSim/vggface1way3shot4layersRandomTrainRandomTest/2connection/saved_models/train_LOSS_z2vae0_z20_g1.0_d0.1_kl0.0_cla1.0_fzl_cla0.0_reconsB1.0_matchingG0.0_matchingD1000.0_sim0.0_Net_batchsize32_zdim256_epoch25.ckpt --restore_classifier_path




## 20 way 5 shot
CUDA_VISIBLE_DEVICES=1 python train_dagan_augmented_general_fewshot_classification_with_matchingclassifier.py --dataset omniglot --image_width 28 --batch_size 32 --generator_inner_layers 3 --discriminator_inner_layers 3 --augmented_number 1024 --is_fewshot_setting 1 --experiment_title AugmentedClassification_omniglot --num_of_gpus 1 --z_dim 256 --dropout_rate_value 0 --selected_classes 20 --support_number 3 --matching 1 --fce 0 --full_context_unroll_k 0 --average_per_class_embeddings 0 --general_classification_samples 5 --pretrained_epoch 0 --classification_total_epoch 60 --restore_path ./RandomSim/omniglot1way3shot4layersRandomTrainRandomTest/2connection/saved_models/train_LOSS_z2vae0_z20_g1.0_d0.1_kl0.0_cla1.0_fzl_cla0.0_reconsB0.0_matchingG0.0_matchingD1.0_sim0.0_Net_batchsize32_zdim256_epoch115.ckpt --restore_classifier_path








