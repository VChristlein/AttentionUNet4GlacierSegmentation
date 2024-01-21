S0_data_info.py (Datagen):
	@params:
		--patch_size [int] (spatial dimemension of a patch)
		--outpath [str] (output folder of the data)
		--bilateral (use bilateral filter on SAR)
		--clahe (use CLAHE on SAR)
	 	--median (use median filtering)
		--padding (use zero-padding to form squares)
		--thicken (use morphological line thickening)
		--nopatch (dont generate patches / one image)

	@output: generates the train/val/test data in @outpath



main.py (Train the network)	

	@params:
		--epochs [int] (number of max epochs)
		--batch_size [int] (Batch size)
		--patch_size [int] (Size of the image)	
		--early_stopping (use stopping criteroin - 20 epochs)
		--cyclic (use cylic learning rate)
		--outpath [str] (input/output path for the data (generated by S0_data_info.py))
		--distance_weight [0/4/8/16] (Use distance weighted loss/metrics)
		--attention (Use Attention u-Net for training)
	
	@output: generates best model.hdf5 and other saved weights in @outpath/test/$timestamp$/ + loss plots


testing.py (Evaluation of the test set (qualitative and quantitative results))

	@params:
		--patch_size [int] (Size of the image)	
		--modelpath [str] (path to the model)
		--datapath [str] (path to the test data (normally 'datafolder/test'))
		--outpath [str] (outpath of the quali./quanti. results )
		--padding (Usage of additional zero padding to the test images)
		--nopatch (Do not use patches)
		--lines (use glacier fronts as gt)
		--nothreshold (Do not search for the optimal binary decision boundary)
		--attention (Test an Attention U-Net)
		--distance_weight [0/4/8/16] (Distance weighted loss on which the model is trained on)
	
	@output: Quantitative reusults (ReportOnModel.txt) + predictions in @outpath

AttentionVisualize (Plot attention maps on each layer) - Only works for Attention U-Net
	

	@params:
		--modeldir [str] (Directory of the saved models)
		--testimg [str] (Path to the test image)
		--testimg_gt [str] (Path to the ground truth of the image)
	
	@output: plots 4 attention maps for epoch 5, 10, 50 and the best epoch for @testimg


A example pipeline would be:

	- python3 S0_data_info.py --outpath data_512_median_nopatch --median --padding --thicken --patch_size 512 --nopatch
	- python3 main.py --outpath data_512_median_nopatch --patch_size 512 --epochs 300 --batch_size 5 --early_stopping --cyclic --distance_weight 4 --attention
	- python3 testing.py --modelpath data_512_median_nopatch/test/$timestamp$/model.hdf5 --datapath data_512_median_nopatch/test/ --outpath data_512_median_nopatch/test/$timestamp$ --nopatch --patch_size 512 --lines --attention --distance_weight 4
	- python3 AttentionVisualize.py --modeldir data_512_median_nopatch/test/$timestamp$/ --testimg data_512_median_nopatch/test/images/2007-01-01_RSAT_20_3.png --testimg_gt data_512_median_nopatch/test/masks_lines/2007-01-01_RSAT_20_3_front.png
