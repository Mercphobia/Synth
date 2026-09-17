# Homebrew formula for Synth
# Install: brew install synth
#
# Note: This is a template. Update version and sha256 for actual releases.

class Synth < Formula
  desc "CLI AI agent framework with local model support"
  homepage "https://github.com/Mercphobia/Synth"
  url "https://github.com/Mercphobia/Synth/archive/refs/tags/v0.1.0.tar.gz"
  sha256 "REPLACE_WITH_ACTUAL_SHA256"
  license "MIT"

  depends_on "python@3.12"

  def install
    # Create virtual environment
    venv = libexec/"venv"
    system "python3.12", "-m", "venv", venv
    
    # Install synth into venv
    system venv/"bin/pip", "install", "-U", "pip"
    system venv/"bin/pip", "install", "."
    
    # Create wrapper script
    (bin/"synth").write <<~EOS
      #!/bin/bash
      exec "#{venv}/bin/synth" "$@"
    EOS
    
    chmod 0755, bin/"synth"
  end

  test do
    system "#{bin}/synth", "--version"
  end
end
