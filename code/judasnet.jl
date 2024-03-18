JudasNET(n_in::Int, n_out::Int) = Chain(
    RLSTM(n_in => 100),
    RepeatVector(n_out),
    RLSTM(100 => 100),
    TimeDistributed(Dense(100, n_out))
)