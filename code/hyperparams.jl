Base.@kwdef mutable struct Args
    η::Float32 = 1.0f-3        
    λ::Float32 = 1.0f-2        
    batch_size::Int = 128      
    sample_size::Int = 128     
    seed::Int = 0              
    cuda::Bool = true          
    epochs::Int = 50           
    val_split::Float32 = 0.8f0 
    early_stop::Int = 5        
    show_plot::Bool = true     
    save_path::String = "temp" 
    best_path::String = "best" 
    #device::Function = gpu
end
