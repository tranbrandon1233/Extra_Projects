import java.util.concurrent.*;

public class RequestGateway {
    private final ConcurrentHashMap<String, Integer> requestCounts = new ConcurrentHashMap<>();
    private final ConcurrentSkipListSet<String> blockedIPs = new ConcurrentSkipListSet<>();
    private final ScheduledExecutorService scheduler = Executors.newScheduledThreadPool(1);

    public void handleRequest(String ipAddress) {
        if (blockedIPs.contains(ipAddress)) {
            System.out.println("Request blocked for IP: " + ipAddress);
            return;
        }

        requestCounts.merge(ipAddress, 1, Integer::sum); // Increment request count

        if (requestCounts.get(ipAddress) > MAX_REQUESTS) {
            blockIP(ipAddress);
        } else {
            // Process the request normally
            System.out.println("Request processed for IP: " + ipAddress);
        }
    }

    private void blockIP(String ipAddress) {
        blockedIPs.add(ipAddress);
        System.out.println("IP blocked: " + ipAddress);

        // Schedule unblock after a certain period 't'
        scheduler.schedule(() -> unblockIP(ipAddress), t, TimeUnit.SECONDS);
    }

    private void unblockIP(String ipAddress) {
        blockedIPs.remove(ipAddress);
        requestCounts.remove(ipAddress);
        System.out.println("IP unblocked: " + ipAddress);
    }

    // Main method or other methods to initialize and manage the gateway
    int MAX_REQUESTS = 100;
    int t = 60;

    public static void main(String[] args) {

        RequestGateway gateway = new RequestGateway();
        gateway.handleRequest("192.1.1.1");
        gateway.blockIP("192.1.1.1");
        gateway.handleRequest("192.1.1.1");
        gateway.unblockIP("192.1.1.1");
        gateway.handleRequest("192.1.1.1");
    }
}
