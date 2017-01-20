HOME='../';
HOMEIMAGES = [HOME 'Images'];
HOMEGIST = [HOME 'Gist'];
HOMEGIST_TEST = [HOME 'Gist_test'];


priorModel = [HOME 'models/priorModel'];
measurementModel = [HOME 'models/measurementModel'];
gistPredictions = [HOME 'models/gistPredictions'];
ContextResults = [HOME 'models/ContextResults'];
performances=[HOME 'models/performances'];
set(0, 'DefaultAxesFontName', 'Helvetica')
set(0,'defaultTextFontName', 'Helvetica')
set(0,'DefaultAxesFontSize',13)
set(0,'DefaultTextFontSize',13)

addpath(genpath(HOME));
doviz=0; % Plot chow-liu's tree & save figures
dosave=0; % Save the models & results
parts=[1 2 3];
for part=parts
	if part==1
		if ~exist('train_imdb')
			disp('loading training set...') 
			load('sun09_groundTruth','Dtraining')
			train_imdb=Dtraining;
			clear Dtraining
			load('sun09_objectCategories');
		end

		N = length(names); % #Categories
		M = length(train_imdb); % #Images

		%% --------------Context model
		dependencies
		location
		%--------------Measurement model
		measurement

		if ~exist('test_imdb')
			disp('loading test set...') 
			load('sun09_groundTruth','Dtest');
			test_imdb=Dtest;
			clear Dtest
			load('sun09_objectCategories');
		end

		gist

		%---------------Save learnt models
		if dosave
		disp('Saving the models..')
			save(priorModel,'A','node_potential', 'edge_potential','prob_bi1',...
		    'root','N','edge_weight','locationPot','edges');

			save(measurementModel,'windowScore','detectionWindowLoc')

			save(gistPredictions,'p_b_gist_test','class_training','class_test')
		end
	elseif part==2
		%% ---------------------------------------------------------------
		if ~exist('test_imdb')
			disp('loading test set...') 
			load('sun09_groundTruth','Dtest');
			test_imdb=Dtest;
			clear Dtest
			load('sun09_objectCategories');
		end
		if ~exist('edges')
			disp('Loading pre-trained models')
			load(priorModel)
			load(measurementModel)
			load(gistPredictions)
		end
		clear train_imdb
		%--------------Alternating inference on trees:
		alternate
		if dosave
			disp('Saving the results..')
			save(ContextResults,'DdetectortestUpdate','presence_truth','presence_score_c','presence_score');
		end
		%--------------Assess performance and plot figures for report/poster:
	else
		if ~exist('test_imdb')
			disp('loading test set...') 
			load('sun09_groundTruth','Dtest');
			test_imdb=Dtest;
			clear Dtest
			load('sun09_objectCategories');
			disp('Loading the test results')
			load(ContextResults)
			load(gistPredictions,'class_test')
		end
		
		perf
		if dosave
			save(performances,'names','ap','ap_context','ap_presence_context','ap_presence')
		end

	end
end