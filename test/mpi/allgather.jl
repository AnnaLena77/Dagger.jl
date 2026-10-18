# `Dagger.allgather`: every rank gets the values of tasks owned by all ranks, without a gather task or a broadcast,
# and without taking task or reference ids.
ag_make(i::Int, n::Int) = fill(Float64(i), n)
ag_sum(x::Vector{Float64}) = sum(x)
ag_bad(i::Int) = i > 0 ? error("allgather failure $i") : i

@testset "Dagger.allgather" begin
    last_thread(r) = Dagger.ExactScope(last(sort(filter(p -> p.rank == r && p.innerProc isa Dagger.ThreadProc, mpi_procs());
                                                  by=p -> p.innerProc.tid)))
    Dagger.check_uniformity!(false)
    # The number of partners per round decides how many rounds a rank takes; every fanout must give the same answer
    @testset "native collective" begin
        MPIExt.MPI_ALLGATHER_NATIVE[] = true
        Dagger.elide_foreign_tasks!(true)
        n = 2 * nranks + 1
        ts = [Dagger.@spawn(scope=last_thread(mod(i, nranks)), return_type=Vector{Float64}, ag_make(i, 50 + i)) for i in 1:n]
        vals = Dagger.allgather(ts)
        @test all(i -> vals[i] == fill(Float64(i), 50 + i), 1:n)
        # sizes differ a lot between ranks, some are empty
        mixed = [Dagger.@spawn(scope=last_thread(mod(i, nranks)), return_type=Vector{Float64}, ag_make(i, i == 1 ? 0 : 20_000 * i)) for i in 1:n]
        mv = Dagger.allgather(mixed)
        @test all(i -> length(mv[i]) == (i == 1 ? 0 : 20_000 * i), 1:n)
        Dagger.elide_foreign_tasks!(false)
    end
    # The rounds of the hand-written version: the number of partners per round decides how many rounds a rank takes;
    # every fanout must give the same answer
    @testset "rounds, fanout $k" for k in (1, 2, 3, 4)
        MPIExt.MPI_ALLGATHER_NATIVE[] = false
        MPIExt.MPI_ALLGATHER_FANOUT[] = k
        Dagger.elide_foreign_tasks!(true)
        n = 2 * nranks + 1
        ts = [Dagger.@spawn(scope=last_thread(mod(i, nranks)), return_type=Vector{Float64}, ag_make(i, 50 + i)) for i in 1:n]
        vals = Dagger.allgather(ts)
        @test all(i -> vals[i] == fill(Float64(i), 50 + i), 1:n)
        Dagger.elide_foreign_tasks!(false)
    end
    MPIExt.MPI_ALLGATHER_FANOUT[] = 4
    MPIExt.MPI_ALLGATHER_NATIVE[] = nothing          # the default again (native, unless the environment says otherwise)
    for elide in (true, false)
        Dagger.elide_foreign_tasks!(elide)
        try
            @testset "owners spread over the ranks, elision=$elide" begin
                n = 2 * nranks + 1                       # uneven: some ranks own more tasks than others
                ts = [Dagger.@spawn(scope=last_thread(mod(i, nranks)), return_type=Vector{Float64}, ag_make(i, 100 + 7i)) for i in 1:n]
                vals = Dagger.allgather(ts)
                @test length(vals) == n
                @test all(i -> vals[i] == fill(Float64(i), 100 + 7i), 1:n)
                # the order of the list decides the order of the result, whatever the owners are
                vals2 = Dagger.allgather(reverse(ts))
                @test vals2 == reverse(vals)
            end
            @testset "empty list and a large payload, elision=$elide" begin
                @test Dagger.allgather(Dagger.DTask[]) == []
                big = [Dagger.@spawn(scope=last_thread(mod(i, nranks)), return_type=Vector{Float64}, ag_make(i, 1_000_000)) for i in 1:nranks]
                vals = Dagger.allgather(big)
                @test all(i -> length(vals[i]) == 1_000_000 && vals[i][end] == Float64(i), 1:nranks)
            end
            @testset "a failed task reaches every rank, elision=$elide" begin
                ok = Dagger.@spawn scope=last_thread(0) return_type=Int ag_bad(0)
                bad = Dagger.@spawn scope=last_thread(min(1, nranks - 1)) return_type=Int ag_bad(1)
                @test_throws Exception Dagger.allgather([ok, bad])
                # and the next call still works, ranks are in step
                @test Dagger.allgather([ok]) == [0]
            end
            @testset "ids stay in agreement, elision=$elide" begin
                before = Dagger.@spawn scope=last_thread(0) return_type=Int ag_bad(0)
                fetch(before)
                Dagger.allgather([before])
                t = Dagger.@spawn scope=last_thread(nranks - 1) return_type=Int ag_bad(0)
                @test fetch(t) == 0
                @test allequal(MPI.Allgather(Int(t.uid), comm))
            end
        finally
            Dagger.elide_foreign_tasks!(false)
        end
    end
    Dagger.check_uniformity!(true)
    MPI.Barrier(comm)
end
