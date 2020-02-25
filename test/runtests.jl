using RobustFactorizations, ProximalOperators
using Test
using Random, Statistics, LinearAlgebra

@testset "RobustFactorizations.jl" begin

    @testset "Robust PCA" begin
        @info "Testing Robust PCA"

        @testset "Nonneg" begin
            @info "Testing Nonneg"

            D = [0.462911    0.365901  0.00204357    0.692873    0.935861;
            0.0446199    0.108606   0.0664309   0.0736707    0.264429;
            0.320581    0.287788    0.073133    0.188872    0.526404;
            0.356266    0.197536 0.000718338    0.513795    0.370094;
            0.677814    0.011651    0.818047   0.0457694    0.471477]

            A = [0.462911   0.365901  0.00204356   0.345428   0.623104;
            0.0446199  0.108606  0.0429271    0.0736707  0.183814;
            0.320581   0.203777  0.073133     0.188872   0.472217;
            0.30725    0.197536  0.000717701  0.201626   0.370094;
            0.234245   0.011651  0.103622     0.0457694  0.279032]

            E = [0.0        0.0        0.0        0.347445  0.312757 ;
            0.0        0.0        0.0235038  0.0       0.0806151;
            0.0        0.0840109  0.0        0.0       0.0541868;
            0.0490157  0.0        6.5061e-7  0.312169  0.0      ;
            0.443569   0.0        0.714425   0.0       0.192445]

            res = rpca(D, nonnegS=true, nonnegL=true, verbose=false)

            @test res.L ≈ A atol=1e-6
            @test res.S ≈ E atol=1e-6
            @test norm(D - (res.L + res.S))/norm(D) < 1e-6


            res = rpca(D, nonnegS=false, nonnegL=false, verbose=false)
            @test norm(D - (res.L + res.S))/norm(D) < 1e-6
        end


        @testset "Complex" begin
            @info "Testing Complex"
            u = randn(ComplexF64, 100)
            v = randn(ComplexF64, 20)
            E = randn(ComplexF64,100,20) .* 10 .* (rand.() .< 0.01)
            A = u*v'
            D = A .+ E
            res = rpca(D)
            @test sum(abs2, res.S-E)/sum(abs2, E) < 1e-5
            @test sum(abs2, res.L-A)/sum(abs2, A) < 1e-5
        end

        @testset "ElasticNet" begin
            @info "Testing ElasticNet"

            # Dense and sparse noise
            D = lowrank(100,10,3)
            E = 0.1randn(100,10) + sparserandn(100,10)
            Dn = D + E
            λ = 1/sqrt(100)
            res1 = rpca(Dn, verbose=false)
            res2 = rpca(Dn, verbose=false, proxD=SqrNormL2(0.5))
            @test norm(Dn - (res1.L + res1.S + res1.D))/norm(D) < 1e-7
            @show norm(D - res1.L) > norm(D - res2.L) # This test was not robust

            N = 1000; r=4;s=1e-1
            for N in [100,1000,10_000], r in [2,4,6]
                D = lowrank(N,10,r)
                for s in [1e-3, 1e-2, 1e-1]
                    @show N,r,s
                    λ = 1/sqrt(N)

                    # Only sparse noise
                    E = 10sparserandn(N,10)
                    Dn = D + E
                    res1 = rpca(Dn, verbose=false)
                    res2 = rpca(Dn, verbose=false, proxD=SqrNormL2(λ/(s)))
                    @test norm(D - res1.L) < norm(D - res2.L)

                    # Dense and large sparse noise
                    E .+= s*randn(N,10)
                    Dn = D + E
                    res1 = rpca(Dn, verbose=false)
                    res2 = rpca(Dn, verbose=false, proxD=SqrNormL2(λ/(s)))
                    @show norm(D - res1.L)
                    @show norm(D - res2.L)

                    # @test norm(D - Â1) > norm(D - Â2)
                end
            end
        end
    end



    @testset "Robust PCA FISTA" begin
        @info "Testing Robust PCA FISTA"

        @testset "Nonneg" begin
            @info "Testing Nonneg"

            D = [0.462911    0.365901  0.00204357    0.692873    0.935861;
            0.0446199    0.108606   0.0664309   0.0736707    0.264429;
            0.320581    0.287788    0.073133    0.188872    0.526404;
            0.356266    0.197536 0.000718338    0.513795    0.370094;
            0.677814    0.011651    0.818047   0.0457694    0.471477]

            A = [0.462911   0.365901  0.00204356   0.345428   0.623104;
            0.0446199  0.108606  0.0429271    0.0736707  0.183814;
            0.320581   0.203777  0.073133     0.188872   0.472217;
            0.30725    0.197536  0.000717701  0.201626   0.370094;
            0.234245   0.011651  0.103622     0.0457694  0.279032]

            E = [0.0        0.0        0.0        0.347445  0.312757 ;
            0.0        0.0        0.0235038  0.0       0.0806151;
            0.0        0.0840109  0.0        0.0       0.0541868;
            0.0490157  0.0        6.5061e-7  0.312169  0.0      ;
            0.443569   0.0        0.714425   0.0       0.192445]

            res = rpca_fista(D, nonnegS=true, nonnegL=true, verbose=false)

            @test norm(res.L-A)/norm(A) < 0.9
            @test norm(res.S-E)/norm(E) < 0.9
            @test norm(D - (res.L + res.S))/norm(D) < 0.9


            res = rpca_fista(D, nonnegS=false, nonnegL=false, verbose=false, iters=50000, λ=0.2)
            @test norm(res.L-A)/norm(A) < 0.9
            @test norm(res.S-E)/norm(E) < 0.9
        end


        @testset "Complex" begin
            @info "Testing Complex"
            u = randn(ComplexF64, 100)
            v = randn(ComplexF64, 20)
            E = randn(ComplexF64,100,20) .* 10 .* (rand.() .< 0.01)
            A = u*v'
            D = A .+ E
            res = rpca_fista(D)
            @test sum(abs2, res.S-E)/sum(abs2, E) < 1e-3
            @test sum(abs2, res.L-A)/sum(abs2, A) < 1e-3
        end
    end



    @testset "Robust PCA ADMM" begin
        @info "Testing Robust PCA ADMM"

        @testset "Nonneg" begin
            @info "Testing Nonneg"

            D = [0.462911    0.365901  0.00204357    0.692873    0.935861;
            0.0446199    0.108606   0.0664309   0.0736707    0.264429;
            0.320581    0.287788    0.073133    0.188872    0.526404;
            0.356266    0.197536 0.000718338    0.513795    0.370094;
            0.677814    0.011651    0.818047   0.0457694    0.471477]

            A = [0.462911   0.365901  0.00204356   0.345428   0.623104;
            0.0446199  0.108606  0.0429271    0.0736707  0.183814;
            0.320581   0.203777  0.073133     0.188872   0.472217;
            0.30725    0.197536  0.000717701  0.201626   0.370094;
            0.234245   0.011651  0.103622     0.0457694  0.279032]

            E = [0.0        0.0        0.0        0.347445  0.312757 ;
            0.0        0.0        0.0235038  0.0       0.0806151;
            0.0        0.0840109  0.0        0.0       0.0541868;
            0.0490157  0.0        6.5061e-7  0.312169  0.0      ;
            0.443569   0.0        0.714425   0.0       0.192445]

            res = rpca_admm(D, nonnegS=true, nonnegL=true, verbose=true)

            @test norm(res.L-A)/norm(A) < 0.2
            @test norm(res.S-E)/norm(E) < 0.2
            @test norm(D - (res.L + res.S))/norm(D) < 1e-8


        end


        @testset "Complex" begin
            @info "Testing Complex"
            u = randn(ComplexF64, 100)
            v = randn(ComplexF64, 20)
            E = randn(ComplexF64,100,20) .* 10 .* (rand.() .< 0.01)
            A = u*v'
            D = A .+ E
            res = rpca_admm(D,ρ=50, tol=1e-5, iters=5000)
            @test sum(abs2, res.S-E)/sum(abs2, E) < 0.01
            @test sum(abs2, res.L-A)/sum(abs2, A) < 0.01
        end
    end


end
