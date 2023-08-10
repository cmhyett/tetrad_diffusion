using DifferentialEquations, DiffEqSensitivity;
using Flux, DiffEqFlux;
using Test, OrdinaryDiffEq, CuArrays, Statistics;
using Optim, LinearAlgebra;


function tetradLearning()
    #construct initial conditions:
    u0 = rand(Float32, 18); #unsure what are reasonable initial conditions are.
    #create random position vectors and combine to form ρ
    # r_1 = rand(Float32, 3);
    # r_2 = rand(Float32, 3);
    # r_3 = rand(Float32, 3);
    # r_4 = rand(Float32, 3);
    r_1 = [1.0, 0.0, 0.0];
    r_2 = [0.0, 1.0, 0.0];
    r_3 = [0.0, 0.0, 1.0];
    r_4 = [1.0, 1.0, 1.0];
    ρ_1 = (1/sqrt(2))*(r_1 - r_2);
    ρ_2 = (1/sqrt(6))*(r_1 + r_2 - 2r_3);
    ρ_3 = (1/sqrt(12))*(r_1 + r_2 + r_3 - 3r_4);
    ρ = [ρ_1 ρ_2 ρ_3];
    k = inv(ρ);
    Π = (k*k')/tr(k*k');
    M = rand((-1.0:0.1:1.0),(3,3));
    M[3,3] = -(M[1,1] + M[2,2]);
    u = [ρ; M];
    u0 = [reshape(u[1:3,1:3],(9,1));reshape(u[4:6,1:3],(9,1))];

    datasize = 100;
    tspan = (0.0f0,0.5f0); #again, unsure what time-scales we're interested in
                           #here or how quickly the dynamics will evolve

    function trueODEfunc(du,u,p,t)
        ρ = reshape(u[1:9],(3,3));
        M = reshape(u[10:18],(3,3));
        #k = inv(ρ);
        #    Π = (k*k')/tr(k*k');
        Π = (1.0/3.0)*Matrix{Float64}(I,3,3)
        placeHolder = [(M'*ρ); (-M^2 + tr(M^2)*Π)];
        du .= [reshape(placeHolder[1:3,1:3],(9,1)); reshape(placeHolder[4:6,1:3],(9,1))] #du/dt = [ dρdt; dMdt]
        #here I neglect the stochastic portion
        # as a first, simplifying step.
    end
    t = range(tspan[1],tspan[2],length=datasize)
    prob = ODEProblem(trueODEfunc,u0,tspan)
    ode_data = Array(solve(prob,AutoTsit5(Rosenbrock23()),saveat=t))

    dudt2 = FastChain(FastDense(18,35,tanh),
                      FastDense(35,35,tanh),
                      FastDense(35,35,tanh),
                      FastDense(35,35,tanh),
                      FastDense(35,18))

    n_ode = NeuralODE(dudt2,tspan,AutoTsit5(Rosenbrock23()),saveat=t)

    function predict_n_ode(p)
        n_ode(u0,p)
    end

    function loss_n_ode(p)
        pred = predict_n_ode(p)
        loss = sum(abs2,ode_data .- pred)
        loss,pred
    end

    closeall();
    #    cb = function (p,l,pred;doplot=false) #callback function to observe training
    cb = function (p,l,pred;doplot=false) #callback function to observe training
        display(l)
        # plot current prediction against data
        if doplot
            p = Array{Plots.Plot{Plots.GRBackend}}(undef, 18)
            for i in 1:18
                p[i] = scatter(t,ode_data[i,1,:],label="data")
                scatter!(p[i],t,pred[i,1,:],label="prediction",legend=:bottomleft)
            end
            #display(plot(p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],p[9],layout=(3,3),legend=false,size=(1000,1000)));
            display(plot(p[10],p[11],p[12],p[13],p[14],p[15],p[16],p[17],p[18],layout=(3,3),legend=false,size=(1000,1000)));
        end
        return false
    end

    # Display the ODE with the initial parameter values.
    cb(n_ode.p,loss_n_ode(n_ode.p)...)

    res1 = DiffEqFlux.sciml_train(loss_n_ode, n_ode.p, ADAM(), cb = cb, maxiters = 5000)
    # cb(res1.minimizer,loss_n_ode(res1.minimizer)...)
    #res2 = DiffEqFlux.sciml_train(loss_n_ode, res1.minimizer, LBFGS(), cb = cb, maxiters = 20)

    return res1,n_ode,u0
    
end

# function gpuNeuralODE()
#     #in general we want to disallow scalar operations
#     CuArrays.allowscalar(true);
#     u0 = Float32[2.; 0.] |> gpu
#     datasize = 30
#     tspan = (0.0f0,10.0f0)

#     function trueODEfunc(du,u,p,t)
#         du[1] = u[2];
#         du[2] = -0.5*u[1]^2 - 2.0*u[2];
#     end
#     t = range(tspan[1],tspan[2],length=datasize)
#     prob = ODEProblem(trueODEfunc,u0,tspan)
#     ode_data = Array(solve(prob,Tsit5(),saveat=t))

#     dudt = Chain(Dense(2,50,tanh), Dense(50,2)) |> gpu

#     function ODEfunc!(du,u,p,t)
#         du .= dudt(u);
#     end

#     n_ode = NeuralODE(dudt,tspan,Tsit5(),saveat=t)

#     function predict_n_ode(p)
#         Array(n_ode(u0,p))
#     end
    
#     # function predict_n_ode(p)
#     #     pred_prob = ODEProblem(ODEfunc, u0, tspan);
#     #     return pred_sol = solve(pred_prob, BS3(), saveat=range(tspan[1],tspan[2],length=datasize));
#     # end

#     function loss_n_ode(p)
#         pred = predict_n_ode(p)
#         loss = sum(abs,ode_data .- pred)
#         loss,pred
#     end

#     closeall();
#     cb = function (p,l,pred;doplot=true) #callback function to observe training
#         display(l)
#         # plot current prediction against data
#         if doplot
#             pl = scatter(t,ode_data[1,:],label="data")
#             scatter!(pl,t,pred[1,:],label="prediction")
#             display(plot(pl))
#         end
#         return false
#     end

#     # # Display the ODE with the initial parameter values.
#     cb(n_ode.p,loss_n_ode(n_ode.p)...)

#     res1 = DiffEqFlux.sciml_train(loss_n_ode, n_ode.p, ADAM(0.5), cb=cb, maxiters = 100)
# #    cb(res1.minimizer,loss_n_ode(res1.minimizer)...;doplot=true)
# end
