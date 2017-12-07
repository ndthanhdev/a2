import vn.hus.nlp.tokenizer.VietTokenizer;

import java.io.IOException;

import py4j.*;

public class VnTokenizerEntryPoint {

	private VietTokenizer tokenizer;

	public VnTokenizerEntryPoint() {
		this.tokenizer = new VietTokenizer();
	}

	public VietTokenizer getTokenizer() {
		return tokenizer;
	}

	public static void main(String[] args) throws IOException {
		// TODO Auto-generated method stub
		GatewayServer gatewayServer = new GatewayServer(new VnTokenizerEntryPoint(), 23333);
		gatewayServer.start();
		System.out.println("Gateway Server Started");
		System.in.read();
	}

}
