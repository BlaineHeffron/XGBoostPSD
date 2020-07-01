using HDF5;
using XGBoost;
using SparseArrays: sparse

include("CommonFunctions.jl")
const nsamp = 150


function predict(bst,test_x,test_y)
    sptest = sparse(test_x)
    preds = predict(bst, sptest)
    print("test-error=", sum((preds .> 0.5) .!= test_y) / float(size(preds)[1]), "\n")
end

function testData(dirs,ntest,ndim,bst)
    i = 0
    y = zeros(ntest)
    x = zeros(ntest,ndim)
    for inputdir in dirs
        if(i > 0)
            j = 1
            while j <= ntest
                y[j] = i
                j++
            end
        end
        nevents = 0
        for (root, dirs, files) in walkdir(inputdir)
            for file in files
                if endswith(file,"WaveformSim.h5")
                    nm = joinpath(root,file)
                    c = h5open(nm, "r") do fid
                        data = read(fid,"Waveforms")
                        curevt = -1
                        for i in data
                            if i.evt != curevt
                                curevt = i.evt
                                nevents += 1
                                if nevents > ntest
                                    predict(bst,x,y)
                                    nevents = 1
                                end
                            end
                            n = 1
                            for x in i.waveform
                                dmx[nevents,i.det*nsamp+n] = x
                                n+=1
                            end
                        end
                    end
                end
            end
        end
        if nevents < nsamp
            predict(bst,x[1:nevents,:],y[1:nevents])
        else
            predict(bst,x,y)
        end
        i++
    end
end


function main()

    if size(ARGS,1) < 2
        println("usage: julia TrainPSD.jl [<input directory1>, <input directory2>, ...]")
        exit(500)
    end

    nTest = 10000
    ndim = 11*14*2*nsamp
    indirs = ARGS #input directories
    modelname = getName(indirs)
    bst = Booster(model_file = string(modelname,".model"))
    #predict
    testData(indirs,nTest,ndim,bst)

end


@time main()
