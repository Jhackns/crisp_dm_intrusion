import tempfile
from pathlib import Path
import pandas as pd
try:
    import pyshark
except Exception:
    pyshark = None
try:
    from scapy.all import rdpcap, IP, TCP, UDP
except Exception:
    rdpcap = None
    IP = None
    TCP = None
    UDP = None

def _service_from_port(port, proto):
    m = {
        80: "http",
        443: "https",
        21: "ftp",
        25: "smtp",
        23: "telnet",
        53: "domain",
        110: "pop3",
        143: "imap4",
        22: "ssh",
    }
    return m.get(port, f"{proto}_{port}")

def _flag_from_tcp(tcp):
    syn = getattr(tcp, "syn", None)
    ack = getattr(tcp, "ack", None)
    rst = getattr(tcp, "rst", None)
    if rst == "1":
        return "REJ"
    if syn == "1" and ack == "1":
        return "SF"
    if syn == "1" and ack != "1":
        return "S0"
    return "SF"

def extract_features_from_pcap(uploaded_file):
    name = str(getattr(uploaded_file, "name", "")).lower()
    ext = ".pcapng" if name.endswith(".pcapng") else ".pcap"
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = Path(tmp.name)
    if pyshark is None and rdpcap is None:
        try:
            from scapy.all import rdpcap as _rdpcap, IP as _IP, TCP as _TCP, UDP as _UDP
            globals()["rdpcap"] = _rdpcap
            globals()["IP"] = _IP
            globals()["TCP"] = _TCP
            globals()["UDP"] = _UDP
        except Exception:
            pass
    flows = {}
    if pyshark is not None:
        cap = pyshark.FileCapture(str(tmp_path), keep_packets=False)
        for pkt in cap:
            ip = getattr(pkt, "ip", None)
            if ip is None:
                continue
            src = getattr(ip, "src", None)
            dst = getattr(ip, "dst", None)
            proto = getattr(ip, "proto", None)
            length = int(getattr(pkt, "length", "0")) if hasattr(pkt, "length") else 0
            tcp = getattr(pkt, "tcp", None)
            udp = getattr(pkt, "udp", None)
            if tcp is not None:
                dport = int(getattr(tcp, "dstport", "0")) if hasattr(tcp, "dstport") else 0
                sport = int(getattr(tcp, "srcport", "0")) if hasattr(tcp, "srcport") else 0
                service = _service_from_port(dport or sport, "tcp")
                flag = _flag_from_tcp(tcp)
                key = (src, dst, "tcp", dport)
            elif udp is not None:
                dport = int(getattr(udp, "dstport", "0")) if hasattr(udp, "dstport") else 0
                sport = int(getattr(udp, "srcport", "0")) if hasattr(udp, "srcport") else 0
                service = _service_from_port(dport or sport, "udp")
                flag = "SF"
                key = (src, dst, "udp", dport)
            else:
                service = "other"
                flag = "SF"
                key = (src, dst, str(proto), 0)
            if key not in flows:
                flows[key] = {
                    "protocol_type": key[2],
                    "service": service,
                    "flag": flag,
                    "src_bytes": 0,
                    "dst_bytes": 0,
                    "count": 0,
                }
            flows[key]["count"] += 1
            flows[key]["src_bytes"] += length
        cap.close()
    elif rdpcap is not None:
        packets = rdpcap(str(tmp_path))
        for pkt in packets:
            if IP is None or not pkt.haslayer(IP):
                continue
            ip_layer = pkt[IP]
            src = getattr(ip_layer, "src", None)
            dst = getattr(ip_layer, "dst", None)
            length = int(len(pkt))
            if pkt.haslayer(TCP):
                tcp_layer = pkt[TCP]
                dport = int(getattr(tcp_layer, "dport", 0))
                sport = int(getattr(tcp_layer, "sport", 0))
                service = _service_from_port(dport or sport, "tcp")
                flags = getattr(tcp_layer, "flags", 0)
                syn = 1 if int(flags) & 0x02 else 0
                ack = 1 if int(flags) & 0x10 else 0
                rst = 1 if int(flags) & 0x04 else 0
                flag = "REJ" if rst else ("SF" if syn and ack else ("S0" if syn and not ack else "SF"))
                key = (src, dst, "tcp", dport)
            elif pkt.haslayer(UDP):
                udp_layer = pkt[UDP]
                dport = int(getattr(udp_layer, "dport", 0))
                sport = int(getattr(udp_layer, "sport", 0))
                service = _service_from_port(dport or sport, "udp")
                flag = "SF"
                key = (src, dst, "udp", dport)
            else:
                service = "other"
                flag = "SF"
                key = (src, dst, "other", 0)
            if key not in flows:
                flows[key] = {
                    "protocol_type": key[2],
                    "service": service,
                    "flag": flag,
                    "src_bytes": 0,
                    "dst_bytes": 0,
                    "count": 0,
                }
            flows[key]["count"] += 1
            flows[key]["src_bytes"] += length
    else:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError("No hay extractor disponible. Instale pyshark (Wireshark/tshark) o scapy.")
    tmp_path.unlink(missing_ok=True)
    rows = []
    for _, v in flows.items():
        row = {
            "protocol_type": v["protocol_type"],
            "service": v["service"],
            "flag": v["flag"],
            "duration": 0.0,
            "src_bytes": float(v["src_bytes"]),
            "dst_bytes": float(v["dst_bytes"]),
            "land": 0.0,
            "wrong_fragment": 0.0,
            "urgent": 0.0,
            "count": float(v["count"]),
            "srv_count": 0.0,
            "serror_rate": 0.0,
            "srv_serror_rate": 0.0,
            "rerror_rate": 0.0,
            "srv_rerror_rate": 0.0,
            "same_srv_rate": 0.0,
            "diff_srv_rate": 0.0,
            "srv_diff_host_rate": 0.0,
            "dst_host_count": 0.0,
            "dst_host_srv_count": 0.0,
            "dst_host_same_srv_rate": 0.0,
            "dst_host_diff_srv_rate": 0.0,
            "dst_host_same_src_port_rate": 0.0,
            "dst_host_srv_diff_host_rate": 0.0,
            "dst_host_serror_rate": 0.0,
            "dst_host_srv_serror_rate": 0.0,
            "dst_host_rerror_rate": 0.0,
            "dst_host_srv_rerror_rate": 0.0,
            "attack_category": "",
            "attack_name": "",
            "service_name": v["service"],
            "protocol_name": v["protocol_type"],
        }
        rows.append(row)
    return pd.DataFrame(rows)