hu = [800,900,1000];
lr = (0.1);
bs = [100,200];
ep = (250);
mo = (0.);
wp = (0);
dp = (0);
[bestError, error_matrix, bestR, bestH, bestL,...
 bestB, bestE, bestM, bestW, bestD] = trainRBM(hu, lr, bs, ep, mo, wp, dp);

fprintf('\nTerminated best error = %f\n', bestError);
fprintf('\nbestH = %d, bestL = %f, bestB = %d, bestE = %d\n',...
        bestH, bestL, bestB, bestE);

save('bestR.mat', 'bestR');
save('error_matrix.mat', 'error_matrix');
