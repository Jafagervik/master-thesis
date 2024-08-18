mutable struct DASDataFrame
    datapath::String
    rows::Int
    nfiles::Int
    cols::Int
    step::Int
    tstart::DateTime
    meta::Dict
end