import paramiko

def kill_alldb_processes():
    ssh_host = "77.93.155.81"
    ssh_user = "administrator"
    ssh_password = "Marek@0705"
    remote_script_path = "/tmp/kill_sleep.sh"

    shell_script = """#!/bin/bash

# Step 1: Dump all KILL statements into a file
mysql --no-defaults -u root -p'Test1234' -N -B -e \\
"SELECT CONCAT('KILL ', id, ';') FROM information_schema.processlist WHERE user != 'system user' AND command = 'Sleep';" > /tmp/kill_list.sql

# Step 2: Run all KILLs at once in a clean session
mysql --no-defaults -u root -p'Test1234' < /tmp/kill_list.sql

# Step 3: Clean up
rm -f /tmp/kill_list.sql
"""

    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(ssh_host, username=ssh_user, password=ssh_password)

        # Upload and execute
        sftp = ssh.open_sftp()
        with sftp.open(remote_script_path, 'w') as f:
            f.write(shell_script)
        sftp.chmod(remote_script_path, 0o755)
        sftp.close()

        print("✅ Script uploaded, executing...")
        stdin, stdout, stderr = ssh.exec_command(f"bash {remote_script_path}")
        print("STDOUT:\n", stdout.read().decode())
        print("STDERR:\n", stderr.read().decode())

        ssh.exec_command(f"rm -f {remote_script_path}")
        ssh.close()

        print("✅ All `Sleep` sessions killed with real KILLs.")

    except Exception as e:
        print(f"❌ Error: {e}")
