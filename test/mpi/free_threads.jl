# `Dagger.free_mpi_threads!`: an MPI task whose thread nobody asked for runs on any free
# thread of its rank instead of waiting behind a busy one.
hog_busy(seconds::Float64) = (t0 = time(); while time() - t0 < seconds; end; 1)   # does not yield
free_quick(x::Int) = x + 1
free_where() = Threads.threadid()

@testset "Free MPI threads" begin
    if Threads.nthreads() >= 3
        cpus = filter(p -> p.innerProc isa Dagger.ThreadProc, mpi_procs())
        rank0 = sort(filter(p -> p.rank == 0, cpus); by=p -> p.innerProc.tid)
        Dagger.free_mpi_threads!(true)
        try
            # Warm up first: the first fetch of a task compiles for a second on slow nodes, which would be measured otherwise
            foreach(fetch, Dagger.with_options(; scope=Dagger.scope(mpi_rank=0)) do
                [Dagger.@spawn(free_quick(i)) for i in 1:16]
            end)
            foreach(fetch, [Dagger.@spawn(scope=Dagger.ExactScope(p), hog_busy(0.01)) for p in rank0])
            MPI.Barrier(comm)
            # Occupy every thread of rank 0 but the first with a pinned (so bound) task that never yields
            hogs = [Dagger.@spawn(scope=Dagger.ExactScope(p), hog_busy(1.5)) for p in rank0[2:end]]
            sleep(0.3)
            MPI.Barrier(comm)
            t0 = time()
            # Scope = the whole rank, 16 tasks: the scheduler names every thread in turn, busy ones included
            tasks = Dagger.with_options(; scope=Dagger.scope(mpi_rank=0)) do
                [Dagger.@spawn(free_quick(i)) for i in 1:16]
            end
            vals = fetch.(tasks)
            elapsed = time() - t0
            @test vals == [i + 1 for i in 1:16]
            # Without the switch, the tasks the scheduler places on a busy thread wait for the hogs (measured: 3.3 s of 4 s)
            rank == 0 && @test elapsed < 1.0       # the hogs still have over a second to run
            # An explicit thread is still a promise
            where = fetch(Dagger.@spawn scope=Dagger.ExactScope(last(rank0)) free_where())
            @test where == last(rank0).innerProc.tid
            foreach(wait, hogs)
        finally
            Dagger.free_mpi_threads!(false)
        end
    end
    MPI.Barrier(comm)
end
