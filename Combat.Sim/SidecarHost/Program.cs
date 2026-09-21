using Combat.Sim.Common;

namespace Combat.Sim.SidecarHost;

/// <summary>JSON Lines Sidecar 进程入口。</summary>
public static class Program
{
    public static void Main()
    {
        var session = new SidecarSession(RepoRootLocator.Find());
        while (Console.ReadLine() is { } line)
        {
            if (string.IsNullOrWhiteSpace(line))
            {
                continue;
            }

            var response = session.Handle(line);
            if (response is null)
            {
                return;
            }

            Console.WriteLine(response);
            Console.Out.Flush();
        }
    }
}
