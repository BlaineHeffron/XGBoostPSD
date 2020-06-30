using HDF5;
using SparseArrays: sparse

evts_per_type = 1e6 #maximum number of events per particle type
test_evts = 1e5 #number of testing events per type
train_filelist = [] #files used for training
test_filelist = [] #files used for testing

function fillDataArrays(x::Array{Int16,2},y::Array{Int16,2},filelist::Array{String,1},n_evts_per_type::Int32,excludeflist::Array{String,1}=[])
    evtcounter = 0
    n = 1
    i = 0
    for d in indirs
        evtcounter = readDir(d,x,evtcounter,filelist,n_evts_per_type,excludefs)
        #simplistic assignment of 0,1,2,3 etc for different particle types
        while n <= evtcounter
            y[n] = i
            n+=1
        end
        i+=1
    end
end


function readDir(inputdir::String,train_x::Array{Int16,2},n::Int64,fs::Array{String,1},maxevts::Int32,direxclude::Array{String,1}=[])
    #reads all WaveformSim files into Int16 2d array
    nEvts = 0
    for (root, dirs, files) in walkdir(inputdir)
        for file in files
            if endswith(file,"WaveformSim.h5")
                nm = joinpath(root,file)
                if nm in direxclude
                    continue
                end
                nEvts += readHDF(nm,train_x,n,maxevts-nEvts)
                fs += nm
                n += nEvts
                if nEvts >= maxevts
                    return n
                end
            end
        end
    end
    return n
end

function readHDF(fname::String,dmx::Array{Int16,2},offset,maxevts)
    nsamp = 150
    nevents = 0
    c = h5open(fname, "r") do fid
        data = read(fid,"Waveforms")
        #println("the string: \t", typeof(data),"\t",data)
        for i in data
            n = 1
            #note this algorithm assumes the events are sorted by event number
            if i.evt + 1 > nevents
                nevents = i.evt +1
                if nevents > maxevts
                    return maxevts
                end
            end
            for x in i.waveform
                dmx[offset+i.evt+1,i.det*nsamp+n] = x
                n+=1
            end
            #@show i.waveform
        end
    end
    return nevents
end

function main()
    if size(ARGS,1) < 2
        println("usage: julia ReadHDF.jl [<input directory1>, <input directory2>, ...]")
        exit(500)
    end

    indirs = ARGS #input directories
    modelname = ""
    i = 0
    for d in indirs
        if !isdir(d)
            error("Error: argument " + d + " is not a directory")
        else
            if i > 0
                modelname +="_" + last(splitdir(d))
            else
                modelname += last(splitdir(d))
            end
        end
        i+=1
    end

    ndet = 14*11*2
    nsamp = 150
    ntype = len(indirs)

    train_x = zeros(Int16, (evts_per_type*ntype,ndet*nsamp))
    train_y = zeros(UInt8, (evts_per_type*ntype))
    test_x = zeros(Int16, (test_evts*ntype,ndet*nsamp))
    test_y = zeros(UInt8, (test_evts*ntype))

    fillDataArrays(train_x,train_y,train_filelist,evts_per_type)
    fillDataArrays(test_x,test_y,test_filelist,test_evts,train_filelist)

    sptrain = sparse(train_x)
    # alternatively, you can pass parameters in as a map
    param = ["max_depth" => 2,
             "eta" => 1,
             "objective" => "binary:logistic"]
    bst = xgboost(sptrain, num_round, label = train_y, param = param)

    # save model to binary local file
    save(bst, modelname + ".model")

    #predict
    preds = predict(bst, test_x)
    print("test-error=", sum((preds .> 0.5) .!= test_y) / float(size(preds)[1]), "\n")

end
