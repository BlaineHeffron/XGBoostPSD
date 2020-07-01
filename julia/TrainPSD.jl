using HDF5;
using XGBoost;
using SparseArrays: sparse

const evts_per_type = 1000000 #maximum number of events per particle type
const test_evts = 10000 #number of testing events per type
const nsamp = 150 #number of samples used
include("CommonFunctions.jl")

function fillDataArrays(x::Array{UInt16,2},y::Array{UInt8,1},indirs,filelist,n_evts_per_type::Int64,excludeflist=[])
    evtcounter = 0
    n = 1
    i = 0
    for d in indirs
        evtcounter = readDir(d,x,evtcounter,filelist,n_evts_per_type,excludeflist)
        #simplistic assignment of 0,1,2,3 etc for different particle types
        while n <= evtcounter
            y[n] = i
            n+=1
        end
        i+=1
    end
end


function readDir(inputdir::String,train_x::Array{UInt16,2},n::Int64,fs,maxevts::Int64,direxclude=[])
    #reads all WaveformSim files into Int16 2d array
    nEvts = 0
    for (root, dirs, files) in walkdir(inputdir)
        for file in files
            if endswith(file,"WaveformSim.h5")
                nm = joinpath(root,file)
                if nm in direxclude
                    continue
                end
                thisevts = readHDF(nm,train_x,n,maxevts-nEvts)
                nEvts += thisevts
                n += thisevts
                append!(fs,nm)
                if nEvts >= maxevts
                    return n
                end
            end
        end
    end
    return n
end

function readHDF(fname::String,dmx::Array{UInt16,2},offset,maxevts)
    nevents = 0
    c = h5open(fname, "r") do fid
        data = read(fid,"Waveforms")
        #println("the string: \t", typeof(data),"\t",data)
        curevt = -1
        for i in data
            if i.evt != curevt
                curevt = i.evt
                nevents += 1
                if nevents > maxevts
                    nevents -= 1
                    return  #returns to the end of the h5open block
                end
            end
            n = 1
            for x in i.waveform
                dmx[offset+nevents,i.det*nsamp+n] = x
                n+=1
            end
            #@show i.waveform
        end
    end
    return nevents
end

function main()
    train_filelist =[] #files used for training
    test_filelist = [] #files used for testing
    if size(ARGS,1) < 2
        println("usage: julia TrainPSD.jl [<input directory1>, <input directory2>, ...]")
        exit(500)
    end

    indirs = ARGS #input directories
    modelname = getName(indirs)

    ndet = 14*11*2
    ntype = length(indirs)

    train_x = zeros(UInt16, (evts_per_type*ntype,ndet*nsamp))
    train_y = zeros(UInt8, (evts_per_type*ntype))
    test_x = zeros(UInt16, (test_evts*ntype,ndet*nsamp))
    test_y = zeros(UInt8, (test_evts*ntype))

    fillDataArrays(train_x,train_y,indirs,train_filelist,evts_per_type)
    fillDataArrays(test_x,test_y,indirs,test_filelist,test_evts,train_filelist)
    println(string("size of arrays is " , size(train_x)))


    sptrain = sparse(train_x)
    # alternatively, you can pass parameters in as a map
    param = ["max_depth" => 2,
             "eta" => 1,
             "objective" => "binary:logistic"]
    num_round =2
    bst = xgboost(sptrain, num_round, label = train_y, param = param)

    # save model to binary local file
    save(bst, string(modelname , ".model"))

    #predict
    sptest = sparse(test_x)
    preds = predict(bst, sptest)
    print("test-error=", sum((preds .> 0.5) .!= test_y) / float(size(preds)[1]), "\n")

end

@time main()
