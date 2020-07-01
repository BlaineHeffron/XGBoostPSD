using HDF5;
using XGBoost;
using SparseArrays: sparse

include("CommonFunctions.jl")
const nsamp = 150


function getPrediction(bst::Booster,test_x::Array{UInt16,2},test_y::Array{UInt8,1})
    sptest = sparse(test_x)
    preds = predict(bst, sptest)
    err = sum((preds .> 0.5) .!= test_y) / float(size(preds)[1])
    print("test-error=", err, "\n")
    return err
end

function testData(dirs,ntest,ndim,bst)
    i = 0
    y = zeros(UInt8,ntest)
    x = zeros(UInt16,ntest,ndim)
    errsum = 0.0
    ntests = 0
    for inputdir in dirs
        if(i > 0)
            j = 1
            while j <= ntest
                y[j] = i
                j+=1
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
                                    errsum += getPrediction(bst,x,y)
                                    ntests += 1
                                    nevents = 1
                                    x = zeros(UInt16,ntest,ndim)
                                end
                            end
                            n = 1
                            for val in i.waveform
                                x[nevents,i.det*nsamp+n] = val
                                n+=1
                            end
                        end
                    end
                end
            end
        end
        if nevents < nsamp
            errsum += getPrediction(bst,x[1:nevents,:],y[1:nevents])
            ntests += 1
            x = zeros(UInt16,ntest,ndim)
        else
            errsum += getPrediction(bst,x,y)
            ntests += 1
            x = zeros(UInt16,ntest,ndim)
        end
        i+=1
    end
    finalerror = errsum/ntests
    print("Overall error: ",finalerror,"\n")
    return finalerror
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
    err = testData(indirs,nTest,ndim,bst)
end


@time main()
