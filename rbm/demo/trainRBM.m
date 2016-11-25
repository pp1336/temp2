function [bestError, error_matrix, bestR, bestH, bestL,...
          bestB, bestE, bestM, bestW, bestD] = trainRBM(hu, lr,...
                                                        bs, ep,...
                                                        mo, wps, dp)

fprintf('\nRBM large step (Caltech 101 Silhouette 28x28 datastet).\n');

load('caltech101_silhouettes_28_split1.mat');

[~,nVis] = size(train_data);

bestError = inf;
bestR = 0;
bestH = 0;
bestL = 0;
bestB = 0;
bestE = 0;
error_matrix = zeros(length(hu),length(lr),length(bs),length(ep),...
                     length(ep), length(ep), length(ep));

t = 1;
for m = mo
    u = 1;
    for wp = wps
        v = 1;
        for d = dp
            w = 1;
            for h = hu
                x = 1;
                for l = lr
                    y = 1;
                    for b = bs
                        z = 1;
                        for e = ep

                             arch = struct('size', [nVis,h],...
                                           'classifier',true,...
                                           'inputType','binary');
                             arch.opts = {'verbose', 1, ...
                             'lRate', l, ...
                             'momentum', m, ...
                             'nEpoch', e, ...
                             'wPenalty', wp, ...
                             'dropout', d,...
                             'batchSz', b, ...
                             'nGibbs', 1, ...
                             };

                             r = rbm(arch);
                             r = r.train(train_data,single(train_labels));

                             [~,classErr,~] = r.classify(test_data,...
                                                   single(test_labels));
                             error = classErr*100;
                             fprintf('\nCompleted with h = %d, l = %f, b = %d, e = %d\nerror = %f\n',...
                                     h, l, b, e, error);
                             error_matrix(w, x, y, z, t, u, v) = error;
                
                             if error < bestError
                                 bestError = error;                    
                                 bestR = r;
                                 bestH = h;
                                 bestL = l;
                                 bestB = b;
                                 bestE = e;
                                 bestM = m;
                                 bestW = wp;
                                 bestD = d;
                             end
                
                         z = z + 1;
                        end
                        y = y + 1;
                    end
                    x = x + 1;
                end
                w = w + 1;
            end
            v = v + 1;
        end
        u = u + 1;
    end
    t = t + 1;
end

end

